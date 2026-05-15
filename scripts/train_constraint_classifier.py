"""
Training script for the Constraint Classifier model.

The model is a multi-label binary classifier over 10 constraint types.
It consumes the 29-feature vector produced by ConstraintFeatureExtractor,
which combines per-section soft-constraint signals with cross-section
conflict aggregates computed from the full university schedule.

Supported input formats:
  1. JSONL, one record per line, with keys:
        section_schedule  : [6][24][3] int array (target section)
        full_uni_schedule : list of [6][24][3] int arrays (all sections)
        violations        : dict {constraint_name: bool}
  2. JSON dict {"schedules": [...]} where each item has either
     `section_schedule`+`full_uni_schedule` or the legacy `schedule` field.

Records that lack `full_uni_schedule` zero out the cross-section features,
which means hard-conflict constraints (instructor_conflict, room_conflict)
cannot be learned from those rows — the data must include the full
university context for those targets to train.

Usage:
    python scripts/train_constraint_classifier.py
    python scripts/train_constraint_classifier.py data/constraint_data.json
    python scripts/train_constraint_classifier.py data/constraint_data.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    src_path = PROJECT_ROOT / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

import src.config as config
from src.feature_extraction import ConstraintFeatureExtractor
from src.models import ConstraintClassifierModel


CONSTRAINT_LABELS = [
    "instructor_conflict",
    "room_conflict",
    "no_lunch_break",
    "late_classes",
    "excessive_hours",
    "saturday_overload",
    "resource_unavailable",
    "curriculum_conflict",
    "room_capacity",
    "instructor_availability",
]

# Subset that depends on cross-section context — targets F1 >= 0.70
HARD_LABELS = {"instructor_conflict", "room_conflict"}

NUM_LABELS = len(CONSTRAINT_LABELS)
WEIGHTED_F1_TARGET = 0.80
HARD_F1_TARGET = 0.70

# Clamp range for per-label positive weights so a rare class doesn't dominate.
POS_WEIGHT_MIN = 1.0
POS_WEIGHT_MAX = 20.0


# ── Data loading ─────────────────────────────────────────────────────────────
def _iter_records(path: Path):
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return
    if path.suffix.lower() == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)
        return
    doc = json.loads(text)
    if isinstance(doc, list):
        yield from doc
    elif isinstance(doc, dict):
        yield from (doc.get("schedules") or doc.get("data") or [])
    else:
        raise ValueError(f"Unsupported top-level JSON type in {path}")


def _normalize_record(rec: dict) -> dict | None:
    section = rec.get("section_schedule")
    if section is None:
        section = rec.get("schedule")
    if section is None:
        legacy = rec.get("original_schedule")
        if isinstance(legacy, dict):
            section = legacy.get("week_schedule")
        else:
            section = legacy
    if section is None:
        return None

    full_uni = rec.get("full_uni_schedule")
    cross_aggregates = rec.get("cross_section")
    violations = rec.get("violations") or rec.get("constraints") or {}
    if not isinstance(violations, dict):
        return None

    return {
        "section": section,
        "full_uni": full_uni,
        "cross_aggregates": cross_aggregates,
        "violations": violations,
    }


# ── Loss & metrics ──────────────────────────────────────────────────────────
def make_weighted_bce(pos_weights: np.ndarray):
    """Per-label weighted binary cross-entropy.

    pos_weights has shape (NUM_LABELS,). Positive examples for label i are
    multiplied by pos_weights[i]; negatives are unweighted.
    """
    pw = tf.constant(pos_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        per_label = -(
            pw * y_true * tf.math.log(y_pred)
            + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )
        return tf.reduce_mean(per_label)

    return loss


def compile_with_weighted_loss(model: keras.Model, pos_weights: np.ndarray):
    cfg = config.CONSTRAINT_CLASSIFIER_CONFIG
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss=make_weighted_bce(pos_weights),
        metrics=[
            keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.5),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc", multi_label=True),
        ],
    )


def _plot_training_curves(
    history: keras.callbacks.History | None,
    out_path: Path,
    title: str,
    metrics: list[str],
):
    if history is None:
        print("No training history; skipping plot.")
        return
    hist = history.history or {}
    if not hist:
        print("Empty training history; skipping plot.")
        return

    series = ["loss"] + [m for m in metrics if m in hist]
    if not series:
        print("No metrics found in history; skipping plot.")
        return

    n = len(series)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    for idx, metric in enumerate(series):
        ax = axes[idx // cols][idx % cols]
        epochs = range(1, len(hist[metric]) + 1)
        ax.plot(epochs, hist[metric], label="train")
        val_key = f"val_{metric}"
        if val_key in hist:
            ax.plot(epochs, hist[val_key], label="val")
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()

    for idx in range(n, rows * cols):
        fig.delaxes(axes[idx // cols][idx % cols])

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves -> {out_path}")


# ── Trainer ──────────────────────────────────────────────────────────────────
class ConstraintClassifierTrainer:
    def __init__(self, data_path: str | None = None):
        self.data_path = (
            Path(data_path)
            if data_path
            else PROJECT_ROOT / "data" / "constraint_data.json"
        )
        self.extractor = ConstraintFeatureExtractor()
        self.scaler = StandardScaler()
        self.model: keras.Model | None = None
        self.history = None

        np.random.seed(config.RANDOM_SEED)
        tf.random.set_seed(config.RANDOM_SEED)

    # ── data ─────────────────────────────────────────────────────────────────
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Constraint data not found at {self.data_path}")

        features: list[np.ndarray] = []
        labels: list[list[float]] = []
        skipped = 0
        with_context = 0
        with_aggregates = 0
        without_context = 0

        for raw in _iter_records(self.data_path):
            if not isinstance(raw, dict):
                skipped += 1
                continue
            rec = _normalize_record(raw)
            if rec is None:
                skipped += 1
                continue
            if rec["cross_aggregates"] is not None:
                with_aggregates += 1
            elif rec["full_uni"]:
                with_context += 1
            else:
                without_context += 1

            try:
                feats = self.extractor.extract(
                    rec["section"],
                    full_uni_schedule=rec["full_uni"],
                    cross_aggregates=rec["cross_aggregates"],
                )
            except Exception:
                skipped += 1
                continue

            vrow = [
                1.0 if bool(rec["violations"].get(name, False)) else 0.0
                for name in CONSTRAINT_LABELS
            ]
            features.append(feats)
            labels.append(vrow)

        if not features:
            raise ValueError("No valid constraint samples could be parsed.")

        X = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.float32)

        print(f"\nLoaded {len(X)} samples from {self.data_path}")
        print(f"  with precomputed cross_section   : {with_aggregates}")
        print(f"  with full_uni_schedule           : {with_context}")
        print(f"  no cross-section source          : {without_context}")
        if skipped:
            print(f"  skipped malformed records        : {skipped}")
        if without_context and not (with_context or with_aggregates):
            print(
                "  WARNING: no record provided full_uni_schedule or cross_section — "
                "cross-section features are all zero, so hard-conflict labels "
                "(instructor_conflict, room_conflict) cannot be learned from this data."
            )

        positives = y.sum(axis=0).astype(int)
        print("\nPositive labels:")
        for name, count in zip(CONSTRAINT_LABELS, positives):
            tag = "  (hard)" if name in HARD_LABELS else ""
            print(f"  {name:<25}: {count}{tag}")

        return X, y

    def preprocess_data(self, X: np.ndarray, y: np.ndarray):
        strata = np.clip(y.sum(axis=1).astype(int), 0, 3)
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y,
            test_size=1.0 - config.TRAIN_RATIO,
            random_state=config.RANDOM_SEED,
            stratify=strata,
        )
        test_fraction = config.TEST_RATIO / (config.TEST_RATIO + config.VALIDATION_RATIO)
        tmp_strata = np.clip(y_tmp.sum(axis=1).astype(int), 0, 3)
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp,
            test_size=test_fraction,
            random_state=config.RANDOM_SEED,
            stratify=tmp_strata,
        )

        self.scaler.fit(X_train)
        joblib.dump(self.scaler, config.CONSTRAINT_SCALER_PATH)
        X_train = self.scaler.transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ── weights ──────────────────────────────────────────────────────────────
    def positive_weights(self, y_train: np.ndarray) -> np.ndarray:
        positives = y_train.sum(axis=0)
        negatives = y_train.shape[0] - positives
        pw = np.ones(NUM_LABELS, dtype=np.float32)
        mask = positives > 0
        pw[mask] = np.clip(negatives[mask] / positives[mask], POS_WEIGHT_MIN, POS_WEIGHT_MAX)

        print("\nPos-weight per label:")
        for name, w, p in zip(CONSTRAINT_LABELS, pw, positives):
            print(f"  {name:<25}: weight={w:.3f}  positives={int(p)}")
        return pw

    # ── model ────────────────────────────────────────────────────────────────
    def build_model(self, input_dim: int, output_dim: int, pos_weights: np.ndarray):
        self.model = ConstraintClassifierModel.build(
            input_dim=input_dim, num_constraints=output_dim,
        )
        # Override the model's default unweighted BCE with the per-label
        # weighted loss so rare hard-conflict labels get a useful gradient.
        compile_with_weighted_loss(self.model, pos_weights)
        self.model.summary()

    def train(self, X_train, y_train, X_val, y_val):
        cfg = config.CONSTRAINT_CLASSIFIER_CONFIG
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg["early_stopping_patience"],
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1,
            ),
            keras.callbacks.CSVLogger(
                str(config.LOGS_DIR / "constraint_classifier_training.csv")
            ),
            keras.callbacks.ModelCheckpoint(
                str(config.CONSTRAINT_CLASSIFIER_PATH),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
        ]

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            callbacks=callbacks,
            verbose=1,
        )

    # ── eval ─────────────────────────────────────────────────────────────────
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        results = self.model.evaluate(X_test, y_test, verbose=1)
        print("\nTest metrics (Keras):")
        for name, value in zip(self.model.metrics_names, results):
            print(f"  {name}: {value:.6f}")

        y_prob = self.model.predict(X_test, verbose=0)
        y_pred = (y_prob >= 0.5).astype(np.int32)
        y_true = y_test.astype(np.int32)

        print("\nPer-label classification report:")
        print(classification_report(
            y_true, y_pred, target_names=CONSTRAINT_LABELS, zero_division=0,
        ))

        per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        per_label_p = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_label_r = recall_score(y_true, y_pred, average=None, zero_division=0)

        print("Per-label P / R / F1:")
        for name, p, r, f in zip(CONSTRAINT_LABELS, per_label_p, per_label_r, per_label_f1):
            tag = "  (hard)" if name in HARD_LABELS else ""
            print(f"  {name:<25}: P={p:.3f}  R={r:.3f}  F1={f:.3f}{tag}")

        micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        print(f"\nMicro F1   : {micro_f1:.4f}")
        print(f"Macro F1   : {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}  (target >= {WEIGHTED_F1_TARGET})")
        if weighted_f1 < WEIGHTED_F1_TARGET:
            print(f"  Warning: weighted F1 below target {WEIGHTED_F1_TARGET}.")

        print("\nHard-constraint targets (F1 >= {:.2f}):".format(HARD_F1_TARGET))
        for name, f in zip(CONSTRAINT_LABELS, per_label_f1):
            if name in HARD_LABELS:
                status = "OK" if f >= HARD_F1_TARGET else "BELOW TARGET"
                print(f"  {name:<25}: F1={f:.3f}  [{status}]")

        return {
            "micro_f1": float(micro_f1),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "per_label_f1": {n: float(f) for n, f in zip(CONSTRAINT_LABELS, per_label_f1)},
        }

    # ── orchestration ────────────────────────────────────────────────────────
    def run(self):
        print("=" * 70)
        print("CONSTRAINT CLASSIFIER — TRAINING PIPELINE (cross-section features)")
        print("=" * 70)

        X, y = self.load_data()
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_data(X, y)
        pos_weights = self.positive_weights(y_train)
        self.build_model(X_train.shape[1], y_train.shape[1], pos_weights)
        self.train(X_train, y_train, X_val, y_val)
        _plot_training_curves(
            self.history,
            config.LOGS_DIR / "constraint_classifier_training.png",
            "Constraint Classifier Training",
            ["binary_accuracy", "precision", "recall", "auc"],
        )
        metrics = self.evaluate(X_test, y_test)

        print("\n" + "=" * 70)
        print("DONE")
        print("=" * 70)
        print(f"  Model  -> {config.CONSTRAINT_CLASSIFIER_PATH}")
        print(f"  Scaler -> {config.CONSTRAINT_SCALER_PATH}")
        return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train constraint classifier")
    parser.add_argument("data", nargs="?", help="Path to constraint data (.json or .jsonl)")
    args = parser.parse_args()

    ConstraintClassifierTrainer(args.data).run()
