"""
Training script for the Mutation Predictor model.

The model predicts whether applying a mutation operator to a class
schedule will improve, leave unchanged (neutral), or worsen its fitness.

It learns from *delta* features — the difference between the before-mutation
and after-mutation schedules — so it can see what the mutation actually
changed, rather than only the post-mutation state.

Supported input formats:
  1. JSONL, one record per line, with keys:
     before_schedule, after_schedule, mutation_type,
     before_fitness, after_fitness, label
  2. JSON dict {"schedules": [...]} where each item has:
     original_schedule, mutated_schedule, original_fitness,
     mutated_fitness, impact, mutation_info.type

Usage:
    python scripts/train_mutation_predictor.py
    python scripts/train_mutation_predictor.py data/mutation_data.json
    python scripts/train_mutation_predictor.py data/mutation_data.jsonl
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    src_path = PROJECT_ROOT / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

import src.config as config
from src.models import MutationPredictorModel


# ── Label / mutation-type encodings ───────────────────────────────────────────
LABELS = ["improve", "neutral", "worsen"]
LABEL_TO_CLASS = {name: i for i, name in enumerate(LABELS)}

MUTATION_TYPES = [
    "day_swap_timeslots",
    "subject_day_swap",
    "slot_nudge",
    "slot_day_nudge",
]
MUTATION_TYPE_TO_IDX = {t: i for i, t in enumerate(MUTATION_TYPES)}

# Schedule grid (mirrors src/config.py)
N_DAYS = config.N_WEEKLY_SCHOOL_DAYS        # 6 — Monday..Saturday
N_SLOTS = config.N_DAILY_TIME_SLOTS         # 24
SATURDAY_IDX = N_DAYS - 1                   # 5

LUNCH_RANGE = range(8, 12)                  # slots 8..11 inclusive
LATE_RANGE = range(20, N_SLOTS)             # slots 20..23 inclusive

# Label thresholds from delta = after - before
IMPROVE_DELTA = 0.5
WORSEN_DELTA = -0.5

EXPECTED_FEATURE_DIM = 41


# ── Feature extraction ───────────────────────────────────────────────────────
def _as_grid(sched) -> np.ndarray:
    """Coerce a schedule into a (N_DAYS, N_SLOTS, 3) int array.

    Accepts either the nested `[day][slot][attr]` array or a dict that wraps
    it under a `week_schedule` key.
    """
    if isinstance(sched, dict):
        sched = sched.get("week_schedule", sched)
    arr = np.asarray(sched, dtype=np.int32)
    if arr.ndim != 3 or arr.shape[0] != N_DAYS or arr.shape[1] != N_SLOTS:
        raise ValueError(
            f"Schedule shape {arr.shape} != ({N_DAYS}, {N_SLOTS}, *)"
        )
    return arr


def _has_lunch(day: np.ndarray) -> float:
    """True iff at least one slot in [8,11] is empty (subject_id == 0)."""
    return float(np.any(day[LUNCH_RANGE.start:LUNCH_RANGE.stop, 0] == 0))


def _has_late_class(day: np.ndarray) -> float:
    """True iff at least one slot >= 20 is occupied (subject_id > 0)."""
    return float(np.any(day[LATE_RANGE.start:LATE_RANGE.stop, 0] > 0))


def _daily_hours(day: np.ndarray) -> float:
    """Occupied slots / 2 (each slot is 30 min)."""
    return float(np.sum(day[:, 0] > 0)) / 2.0


def _days_with_class(sched: np.ndarray) -> float:
    return float(np.sum(np.any(sched[:, :, 0] > 0, axis=1)))


def extract_delta_features(
    before: np.ndarray,
    after: np.ndarray,
    mutation_type: str,
    before_fitness: float,
    after_fitness: float,
) -> np.ndarray:
    """Build the 41-dim feature vector for one (before, after) pair."""
    feats: list[float] = []

    # 1. Mutation-type one-hot (4)
    one_hot = [0.0] * len(MUTATION_TYPES)
    idx = MUTATION_TYPE_TO_IDX.get(mutation_type)
    if idx is not None:
        one_hot[idx] = 1.0
    feats.extend(one_hot)

    # 2. Fitness deltas (3)
    delta = after_fitness - before_fitness
    feats.extend([before_fitness, after_fitness, delta])

    # 3. Per-day lunch/late class before+after (24)
    for d in range(N_DAYS):
        feats.extend([
            _has_lunch(before[d]),
            _has_lunch(after[d]),
            _has_late_class(before[d]),
            _has_late_class(after[d]),
        ])

    # 4. Daily hours delta (6)
    for d in range(N_DAYS):
        feats.append(_daily_hours(after[d]) - _daily_hours(before[d]))

    # 5. Days-with-class before+after (2)
    feats.extend([_days_with_class(before), _days_with_class(after)])

    # 6. Saturday hours before+after (2)
    feats.extend([
        _daily_hours(before[SATURDAY_IDX]),
        _daily_hours(after[SATURDAY_IDX]),
    ])

    vec = np.asarray(feats, dtype=np.float32)
    assert vec.shape[0] == EXPECTED_FEATURE_DIM, (
        f"feature dim {vec.shape[0]} != {EXPECTED_FEATURE_DIM}"
    )
    return vec


# ── Data loading (supports JSONL and the legacy mutation_data.json) ──────────
def _iter_records(path: Path):
    """Yield dict records from either a JSONL file or a JSON document."""
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
        records = doc.get("schedules") or doc.get("data") or []
        yield from records
    else:
        raise ValueError(f"Unsupported top-level JSON type in {path}")


def _normalize_record(rec: dict) -> dict | None:
    """Map either schema onto common keys; return None if record is unusable."""
    before = rec.get("before_schedule") or rec.get("original_schedule")
    after = rec.get("after_schedule") or rec.get("mutated_schedule")
    if before is None or after is None:
        return None

    mtype = rec.get("mutation_type")
    if mtype is None:
        mtype = (rec.get("mutation_info") or {}).get("type")
    mtype = str(mtype or "").lower()

    before_fit = rec.get("before_fitness", rec.get("original_fitness"))
    after_fit = rec.get("after_fitness", rec.get("mutated_fitness"))
    if before_fit is None or after_fit is None:
        return None

    label = rec.get("label") or rec.get("impact")
    return {
        "before": before,
        "after": after,
        "mutation_type": mtype,
        "before_fitness": float(before_fit),
        "after_fitness": float(after_fit),
        "stored_label": str(label).lower() if label is not None else None,
    }


def _label_from_delta(delta: float) -> str:
    if delta > IMPROVE_DELTA:
        return "improve"
    if delta < WORSEN_DELTA:
        return "worsen"
    return "neutral"


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
class MutationPredictorTrainer:
    def __init__(self, data_path: str | None = None):
        self.data_path = (
            Path(data_path)
            if data_path
            else PROJECT_ROOT / "data" / "mutation_data.json"
        )
        self.scaler = StandardScaler()
        self.model: keras.Model | None = None
        self.history = None
        self.mutation_type_warned: set[str] = set()

        np.random.seed(config.RANDOM_SEED)
        tf.random.set_seed(config.RANDOM_SEED)

    def load_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Mutation data not found at {self.data_path}")

        X: list[np.ndarray] = []
        y: list[int] = []
        skipped = 0
        unknown_types: dict[str, int] = {}

        for raw in _iter_records(self.data_path):
            if not isinstance(raw, dict):
                skipped += 1
                continue
            rec = _normalize_record(raw)
            if rec is None:
                skipped += 1
                continue

            try:
                before = _as_grid(rec["before"])
                after = _as_grid(rec["after"])
            except ValueError:
                skipped += 1
                continue

            label = _label_from_delta(rec["after_fitness"] - rec["before_fitness"])
            class_id = LABEL_TO_CLASS[label]

            mtype = rec["mutation_type"]
            if mtype and mtype not in MUTATION_TYPE_TO_IDX:
                unknown_types[mtype] = unknown_types.get(mtype, 0) + 1

            feats = extract_delta_features(
                before,
                after,
                mtype,
                rec["before_fitness"],
                rec["after_fitness"],
            )
            X.append(feats)
            y.append(class_id)

        if not X:
            raise ValueError("No valid mutation samples could be parsed.")

        X_arr = np.stack(X).astype(np.float32)
        y_arr = np.asarray(y, dtype=np.int32)

        print(f"\nLoaded {len(X_arr)} samples from {self.data_path}")
        if skipped:
            print(f"  Skipped {skipped} malformed records")
        if unknown_types:
            top = sorted(unknown_types.items(), key=lambda kv: -kv[1])[:5]
            print("  Mutation types outside the spec one-hot (zero-encoded):")
            for name, count in top:
                print(f"    {name!r}: {count}")

        counts = np.bincount(y_arr, minlength=3)
        print("Class distribution (from delta rule):")
        for name, idx in LABEL_TO_CLASS.items():
            print(f"  {name:<8}: {counts[idx]}")

        return X_arr, y_arr

    def balance_classes(self, X: np.ndarray, y: np.ndarray):
        """Downsample each class to the minimum class size."""
        rng = np.random.RandomState(config.RANDOM_SEED)
        per_class_idx = [np.where(y == c)[0] for c in range(3)]
        n = min(len(idx) for idx in per_class_idx)
        if n == 0:
            raise ValueError("Cannot balance: at least one class has 0 samples.")

        selected = np.concatenate([
            rng.choice(idx, size=n, replace=False) for idx in per_class_idx
        ])
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
        joblib.dump(self.scaler, config.MUTATION_SCALER_PATH)
        X_train = self.scaler.transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        y_train_oh = to_categorical(y_train, num_classes=3)
        y_val_oh = to_categorical(y_val, num_classes=3)
        y_test_oh = to_categorical(y_test, num_classes=3)

        return X_train, X_val, X_test, y_train_oh, y_val_oh, y_test_oh, y_test

    def build_model(self, input_dim: int):
        self.model = MutationPredictorModel.build(input_dim=input_dim, num_classes=3)
        self.model.summary()

    def train(self, X_train, y_train, X_val, y_val):
        cfg = config.MUTATION_PREDICTOR_CONFIG
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=cfg.get("early_stopping_patience", 20),
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=cfg.get("reduce_lr_patience", 10),
                min_lr=1e-7,
                verbose=1,
            ),
            keras.callbacks.CSVLogger(
                str(config.LOGS_DIR / "mutation_predictor_training.csv")
            ),
            keras.callbacks.ModelCheckpoint(
                str(config.MUTATION_PREDICTOR_PATH),
                monitor="val_accuracy",
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

    def evaluate(self, X_test, y_test_oh, y_test_idx):
        results = self.model.evaluate(X_test, y_test_oh, verbose=1)
        print("\nTest metrics:")
        for name, value in zip(self.model.metrics_names, results):
            print(f"  {name}: {value:.6f}")

        y_pred = np.argmax(self.model.predict(X_test, verbose=0), axis=1)

        print("\nClassification report:")
        print(classification_report(
            y_test_idx, y_pred, target_names=LABELS, zero_division=0,
        ))

        print("Confusion matrix (rows=true, cols=pred):")
        print(confusion_matrix(y_test_idx, y_pred, labels=[0, 1, 2]))

        macro_f1 = f1_score(y_test_idx, y_pred, average="macro", zero_division=0)
        accuracy = float(np.mean(y_pred == y_test_idx))
        print(f"\nMacro F1: {macro_f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}  (target >= 0.65)")
        if accuracy < 0.65:
            print("  Warning: accuracy below 0.65 target.")
        return {"accuracy": accuracy, "macro_f1": macro_f1}

    def run(self):
        print("=" * 70)
        print("MUTATION PREDICTOR — DELTA-FEATURE TRAINING PIPELINE")
        print("=" * 70)

        X, y = self.load_data()
        X, y = self.balance_classes(X, y)

        X_train, X_val, X_test, y_train_oh, y_val_oh, y_test_oh, y_test_idx = (
            self.preprocess_data(X, y)
        )

        self.build_model(input_dim=X_train.shape[1])
        self.train(X_train, y_train_oh, X_val, y_val_oh)
        _plot_training_curves(
            self.history,
            config.LOGS_DIR / "mutation_predictor_training.png",
            "Mutation Predictor Training",
            ["accuracy", "top_2_accuracy"],
        )
        metrics = self.evaluate(X_test, y_test_oh, y_test_idx)

        print("\n" + "=" * 70)
        print("DONE")
        print("=" * 70)
        print(f"  Model  -> {config.MUTATION_PREDICTOR_PATH}")
        print(f"  Scaler -> {config.MUTATION_SCALER_PATH}")
        return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train mutation predictor")
    parser.add_argument("data", nargs="?", help="Path to mutation data (.json or .jsonl)")
    args = parser.parse_args()

    MutationPredictorTrainer(args.data).run()
