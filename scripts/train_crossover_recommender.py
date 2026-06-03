"""
Training script for the Crossover Recommender model.

The model is a binary classifier that predicts whether a (parent1, parent2)
pair is *compatible* for crossover — i.e. whether crossing them will produce
a valid offspring with fitness above ``CROSSOVER_RECOMMENDER_CONFIG['fitness_threshold']``.

The GA uses this prediction to skip incompatible parent pairs instead of
retrying crossover up to 384 times.

Supported input formats:
  1. JSONL, one record per line, with keys:
        parent1            : [6][24][3] int array
        parent2            : [6][24][3] int array
        produced_valid     : 1 if crossover succeeded, 0 otherwise
        offspring_fitness  : float (0.0 if produced_valid == 0)
     Optional parent fitness fields: parent1_fitness, parent2_fitness.
  2. JSON dict {"schedules": [...]} where each record carries the same
     parent arrays plus an `offspring_fitness` (and usually a
     `metadata.parent{1,2}_fitness`). `produced_valid` is inferred as
     `offspring_fitness > 0` when absent.

Usage:
    python scripts/train_crossover_recommender.py
    python scripts/train_crossover_recommender.py data/crossover_data.json
    python scripts/train_crossover_recommender.py data/crossover_data.jsonl
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
    confusion_matrix,
    classification_report,
    roc_auc_score,
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
from src.models import CrossoverRecommenderModel


# ── Schedule constants ───────────────────────────────────────────────────────
N_DAYS = config.N_WEEKLY_SCHOOL_DAYS        # 6
N_SLOTS = config.N_DAILY_TIME_SLOTS         # 24
SATURDAY_IDX = N_DAYS - 1
LUNCH_RANGE = range(8, 12)                  # slots 8..11 inclusive
LATE_THRESHOLD = 20
PREFERRED_MAX_HOURS = 10.0

EXPECTED_FEATURE_DIM = 23

# Targets
AUC_TARGET = 0.75
ACC_TARGET = 0.70

# Scaler is helpful here because some features (fitnesses, slot counts) live on
# very different scales; reuse the project's mutation scaler path for a sibling.
SCALER_PATH = config.MODELS_DIR / "crossover_scaler.joblib"


# ── Schedule parsing ─────────────────────────────────────────────────────────
def _as_grid(sched) -> np.ndarray:
    """Coerce input to a (N_DAYS, N_SLOTS, 3) int array. Returns None on bad shape."""
    if isinstance(sched, dict):
        sched = sched.get("week_schedule") or sched.get("schedule") or sched
    arr = np.asarray(sched, dtype=np.int32)
    if arr.ndim != 3 or arr.shape[0] != N_DAYS or arr.shape[1] != N_SLOTS or arr.shape[2] < 3:
        return None
    return arr[:, :, :3]


# ── Per-parent structural features ──────────────────────────────────────────
def _days_with_class(sched: np.ndarray) -> float:
    return float(np.sum(np.any(sched[:, :, 0] > 0, axis=1)))


def _total_hours(sched: np.ndarray) -> float:
    return float(np.sum(sched[:, :, 0] > 0)) / float(config.N_HOUR_TIME_SLOTS)


def _has_saturday_classes(sched: np.ndarray) -> float:
    return float(np.any(sched[SATURDAY_IDX, :, 0] > 0))


def _lunch_break_days(sched: np.ndarray) -> float:
    """Number of days that have at least one free slot in [8,11]."""
    count = 0
    for d in range(N_DAYS):
        subj = sched[d, :, 0]
        if np.any(subj[LUNCH_RANGE.start:LUNCH_RANGE.stop] == 0):
            count += 1
    return float(count)


def _late_class_count(sched: np.ndarray) -> float:
    return float(np.sum(sched[:, LATE_THRESHOLD:, 0] > 0))


def _days_over_preferred_hours(sched: np.ndarray) -> float:
    """Count of days whose occupied-slot hours exceed PREFERRED_MAX_HOURS."""
    hours_per_day = np.sum(sched[:, :, 0] > 0, axis=1) / float(config.N_HOUR_TIME_SLOTS)
    return float(np.sum(hours_per_day > PREFERRED_MAX_HOURS))


def _parent_structural(sched: np.ndarray) -> list[float]:
    return [
        _days_with_class(sched),
        _total_hours(sched),
        _has_saturday_classes(sched),
        _lunch_break_days(sched),
        _late_class_count(sched),
        _days_over_preferred_hours(sched),
    ]


# ── Pair-level features ─────────────────────────────────────────────────────
def _similarity_features(p1: np.ndarray, p2: np.ndarray) -> list[float]:
    """matching_slot_count, matching_ratio, occupied_slot_overlap."""
    s1 = p1[:, :, 0]
    s2 = p2[:, :, 0]
    occ1 = s1 > 0
    occ2 = s2 > 0

    matching = int(np.sum((s1 == s2) & occ1 & occ2))        # same non-zero subject
    overlap = int(np.sum(occ1 & occ2))                       # both occupied
    union_occupied = int(np.sum(occ1 | occ2))                # either occupied

    ratio = float(matching) / float(union_occupied) if union_occupied > 0 else 0.0
    return [float(matching), ratio, float(overlap)]


def extract_pair_features(
    p1: np.ndarray,
    p2: np.ndarray,
    p1_fitness: float,
    p2_fitness: float,
) -> np.ndarray:
    """Return the 23-dim compatibility feature vector for one pair."""
    feats: list[float] = []

    # 1. Individual fitnesses (2)
    feats.extend([p1_fitness, p2_fitness])

    # 2. Fitness relationship (2)
    feats.extend([
        p1_fitness - p2_fitness,         # signed difference; the model can square it if it wants
        0.5 * (p1_fitness + p2_fitness),
    ])

    # 3. Schedule similarity (3)
    feats.extend(_similarity_features(p1, p2))

    # 4. Per-parent structural features (12)
    s1 = _parent_structural(p1)
    s2 = _parent_structural(p2)
    feats.extend(s1)
    feats.extend(s2)

    # 5. Structural compatibility deltas (4)
    feats.extend([
        s1[0] - s2[0],   # days_with_class
        s1[1] - s2[1],   # total_hours
        s1[3] - s2[3],   # lunch_break_days
        s1[4] - s2[4],   # late_class_count
    ])

    vec = np.asarray(feats, dtype=np.float32)
    assert vec.shape[0] == EXPECTED_FEATURE_DIM, (
        f"feature dim {vec.shape[0]} != {EXPECTED_FEATURE_DIM}"
    )
    return vec


# ── Data loading ────────────────────────────────────────────────────────────
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


def _record_fitness(rec: dict, key: str) -> float | None:
    """Read a parent fitness from the top-level record or its metadata."""
    if key in rec and rec[key] is not None:
        return float(rec[key])
    meta = rec.get("metadata") or {}
    if key in meta and meta[key] is not None:
        return float(meta[key])
    return None


def _label_from_record(
    rec: dict, fitness_threshold: float
) -> tuple[int, float] | None:
    """Returns (label, offspring_fitness) or None if the record is unlabelable."""
    off_fit = rec.get("offspring_fitness")
    produced_valid = rec.get("produced_valid")

    if produced_valid is None and off_fit is None:
        return None

    off_fit = float(off_fit) if off_fit is not None else 0.0
    if produced_valid is None:
        # legacy data: infer from fitness
        produced_valid = 1 if off_fit > 0.0 else 0

    label = 1 if (int(produced_valid) == 1 and off_fit >= fitness_threshold) else 0
    return label, off_fit


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


# ── Trainer ─────────────────────────────────────────────────────────────────
class CrossoverRecommenderTrainer:
    def __init__(self, data_path: str | None = None):
        self.data_path = (
            Path(data_path)
            if data_path
            else PROJECT_ROOT / "data" / "crossover_data.json"
        )
        self.scaler = StandardScaler()
        self.model: keras.Model | None = None
        self.history = None

        np.random.seed(config.RANDOM_SEED)
        tf.random.set_seed(config.RANDOM_SEED)

    def load_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Crossover data not found at {self.data_path}")

        cfg = config.CROSSOVER_RECOMMENDER_CONFIG
        threshold = float(cfg.get("fitness_threshold", 5.0))

        X: list[np.ndarray] = []
        y: list[int] = []
        skipped = 0
        missing_fitness = 0

        for rec in _iter_records(self.data_path):
            if not isinstance(rec, dict):
                skipped += 1
                continue

            p1 = _as_grid(rec.get("parent1"))
            p2 = _as_grid(rec.get("parent2"))
            if p1 is None or p2 is None:
                skipped += 1
                continue

            label_pair = _label_from_record(rec, threshold)
            if label_pair is None:
                skipped += 1
                continue
            label, _ = label_pair

            p1_fit = _record_fitness(rec, "parent1_fitness")
            p2_fit = _record_fitness(rec, "parent2_fitness")
            if p1_fit is None or p2_fit is None:
                missing_fitness += 1
                p1_fit = p1_fit if p1_fit is not None else 0.0
                p2_fit = p2_fit if p2_fit is not None else 0.0

            X.append(extract_pair_features(p1, p2, p1_fit, p2_fit))
            y.append(label)

        if not X:
            raise ValueError("No valid crossover samples could be parsed.")

        X_arr = np.stack(X).astype(np.float32)
        y_arr = np.asarray(y, dtype=np.int32)

        print(f"\nLoaded {len(X_arr)} samples from {self.data_path}")
        if skipped:
            print(f"  Skipped {skipped} malformed records")
        if missing_fitness:
            print(f"  {missing_fitness} records missing parent fitness — defaulted to 0.0")

        pos = int(np.sum(y_arr == 1))
        neg = int(np.sum(y_arr == 0))
        print(f"  Label 1 (compatible, fitness >= {threshold}): {pos}")
        print(f"  Label 0 (incompatible)                       : {neg}")
        return X_arr, y_arr

    def balance_classes(self, X: np.ndarray, y: np.ndarray):
        """Undersample the majority class (typically valid/compatible) to match
        the minority class size — matches the spec's note that the GA produces
        more valid than invalid pairs.
        """
        rng = np.random.RandomState(config.RANDOM_SEED)
        idx_pos = np.where(y == 1)[0]
        idx_neg = np.where(y == 0)[0]
        if len(idx_pos) == 0 or len(idx_neg) == 0:
            raise ValueError("Cannot balance: one of the classes has 0 samples.")
        n = min(len(idx_pos), len(idx_neg))
        sel_pos = rng.choice(idx_pos, size=n, replace=False)
        sel_neg = rng.choice(idx_neg, size=n, replace=False)
        selected = np.concatenate([sel_pos, sel_neg])
        rng.shuffle(selected)
        print(f"\nBalanced dataset: {n} per class -> {len(selected)} total")
        return X[selected], y[selected]

    def preprocess_data(self, X: np.ndarray, y: np.ndarray):
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y,
            test_size=1.0 - config.TRAIN_RATIO,
            random_state=config.RANDOM_SEED,
            stratify=y,
        )
        test_fraction = config.TEST_RATIO / (config.TEST_RATIO + config.VALIDATION_RATIO)
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp,
            test_size=test_fraction,
            random_state=config.RANDOM_SEED,
            stratify=y_tmp,
        )

        self.scaler.fit(X_train)
        joblib.dump(self.scaler, SCALER_PATH)
        X_train = self.scaler.transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def build_model(self, input_dim: int):
        self.model = CrossoverRecommenderModel.build(input_dim=input_dim)
        self.model.summary()

    def train(self, X_train, y_train, X_val, y_val):
        cfg = config.CROSSOVER_RECOMMENDER_CONFIG
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=cfg.get("early_stopping_patience", 20),
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=cfg.get("reduce_lr_patience", 8),
                min_lr=1e-7,
                verbose=1,
            ),
            keras.callbacks.CSVLogger(
                str(config.LOGS_DIR / "crossover_recommender_training.csv")
            ),
            keras.callbacks.ModelCheckpoint(
                str(config.CROSSOVER_RECOMMENDER_PATH),
                monitor="val_auc",
                mode="max",
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

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        results = self.model.evaluate(X_test, y_test, verbose=1)
        print("\nTest metrics (Keras):")
        for name, value in zip(self.model.metrics_names, results):
            print(f"  {name}: {value:.6f}")

        prob = self.model.predict(X_test, verbose=0).reshape(-1)
        pred = (prob >= 0.5).astype(np.int32)

        try:
            auc = roc_auc_score(y_test, prob)
        except ValueError:
            auc = float("nan")
        accuracy = float(np.mean(pred == y_test))

        print("\nClassification report:")
        print(classification_report(
            y_test, pred, target_names=["invalid (0)", "compatible (1)"], zero_division=0,
        ))

        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        print("Confusion matrix (rows=true, cols=pred) [0=invalid, 1=compatible]:")
        print(cm)
        tn, fp, fn, tp = cm.ravel()
        recall_invalid = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        recall_compatible = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        print(f"\nAccuracy             : {accuracy:.4f}  (target >= {ACC_TARGET})")
        print(f"AUC                  : {auc:.4f}  (target >= {AUC_TARGET})")
        print(f"Recall on invalid (0): {recall_invalid:.4f}  "
              f"<- prioritise this: a missed incompatible pair wastes GA cycles")
        print(f"Recall on compatible : {recall_compatible:.4f}")

        if accuracy < ACC_TARGET:
            print(f"  Warning: accuracy below {ACC_TARGET}")
        if not np.isnan(auc) and auc < AUC_TARGET:
            print(f"  Warning: AUC below {AUC_TARGET}")

        return {
            "accuracy": accuracy,
            "auc": float(auc) if not np.isnan(auc) else None,
            "recall_invalid": float(recall_invalid),
            "recall_compatible": float(recall_compatible),
        }

    def run(self):
        print("=" * 70)
        print("CROSSOVER RECOMMENDER — BINARY COMPATIBILITY PIPELINE")
        print("=" * 70)

        X, y = self.load_data()
        X, y = self.balance_classes(X, y)
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_data(X, y)
        self.build_model(X_train.shape[1])
        self.train(X_train, y_train, X_val, y_val)
        _plot_training_curves(
            self.history,
            config.LOGS_DIR / "crossover_recommender_training.png",
            "Crossover Recommender Training",
            ["accuracy", "auc", "precision", "recall"],
        )
        metrics = self.evaluate(X_test, y_test)

        print("\n" + "=" * 70)
        print("DONE")
        print("=" * 70)
        print(f"  Model  -> {config.CROSSOVER_RECOMMENDER_PATH}")
        print(f"  Scaler -> {SCALER_PATH}")
        return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train crossover compatibility classifier")
    parser.add_argument("data", nargs="?", help="Path to crossover data (.json or .jsonl)")
    args = parser.parse_args()

    CrossoverRecommenderTrainer(args.data).run()
