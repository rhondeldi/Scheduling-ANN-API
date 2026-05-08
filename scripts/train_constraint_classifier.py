"""
Training script for the Constraint Classifier model.

Usage:
    python scripts/train_constraint_classifier.py
    python scripts/train_constraint_classifier.py data/constraint_data.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    # ensure `src` package modules that use bare imports (e.g. `import config`) are importable
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


class ConstraintClassifierTrainer:
    def __init__(self, data_path: str | None = None):
        self.data_path = Path(data_path) if data_path else PROJECT_ROOT / "data" / "constraint_data.json"
        self.extractor = ConstraintFeatureExtractor()
        self.scaler = StandardScaler()
        self.model = None
        self.history = None

        np.random.seed(config.RANDOM_SEED)
        tf.random.set_seed(config.RANDOM_SEED)

    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Constraint data not found at {self.data_path}")

        raw = json.loads(self.data_path.read_text(encoding="utf-8-sig"))
        samples = raw.get("schedules", raw.get("data", raw)) if isinstance(raw, dict) else raw
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"No samples found in {self.data_path}")

        def extract_schedule(item):
            """Extract week_schedule from either nested dict or direct array format."""
            if isinstance(item, dict):
                return item.get("week_schedule")
            elif isinstance(item, list):
                return item
            return None

        features: list[np.ndarray] = []
        labels: list[list[float]] = []

        for sample in samples:
            sched_item = sample.get("schedule") or sample.get("original_schedule")
            ws = extract_schedule(sched_item)
            if ws is None:
                continue

            label_source = sample.get("violations") or sample.get("constraints") or sample
            y = [1.0 if bool(label_source.get(name, False)) else 0.0 for name in CONSTRAINT_LABELS]
            features.append(self.extractor.extract({"week_schedule": ws}))
            labels.append(y)

        if not features:
            raise ValueError("No valid constraint samples could be parsed.")

        X = np.array(features, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)
        positives = y.sum(axis=0).astype(int)
        print("\nPositive labels:")
        for name, count in zip(CONSTRAINT_LABELS, positives):
            print(f"  {name:<25}: {count}")
        return X, y

    def preprocess_data(self, X: np.ndarray, y: np.ndarray):
        strata = np.clip(y.sum(axis=1).astype(int), 0, 3)
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X,
            y,
            test_size=1.0 - config.TRAIN_RATIO,
            random_state=config.RANDOM_SEED,
            stratify=strata,
        )
        test_fraction = config.TEST_RATIO / (config.TEST_RATIO + config.VALIDATION_RATIO)
        tmp_strata = np.clip(y_tmp.sum(axis=1).astype(int), 0, 3)
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp,
            y_tmp,
            test_size=test_fraction,
            random_state=config.RANDOM_SEED,
            stratify=tmp_strata,
        )

        scaler_path = config.CONSTRAINT_SCALER_PATH
        self.scaler.fit(X_train)
        joblib.dump(self.scaler, scaler_path)

        X_train = self.scaler.transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def build_model(self, input_dim: int, output_dim: int):
        self.model = ConstraintClassifierModel.build(input_dim=input_dim, num_constraints=output_dim)
        self.model.summary()

    def train(self, X_train, y_train, X_val, y_val, sample_weight):
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
            keras.callbacks.CSVLogger(str(config.LOGS_DIR / "constraint_classifier_training.csv")),
            keras.callbacks.ModelCheckpoint(
                str(config.CONSTRAINT_CLASSIFIER_PATH),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
        ]

        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            callbacks=callbacks,
            sample_weight=sample_weight,
            verbose=1,
        )

    def evaluate(self, X_test, y_test):
        results = self.model.evaluate(X_test, y_test, verbose=1)
        print("\nTest metrics:")
        for name, value in zip(self.model.metrics_names, results):
            print(f"  {name}: {value:.6f}")
        y_prob = self.model.predict(X_test, verbose=0)
        y_pred = (y_prob >= 0.5).astype(np.int32)
        print("\nPer-label report:")
        print(classification_report(y_test.astype(np.int32), y_pred, target_names=CONSTRAINT_LABELS, zero_division=0))
        return results

    def label_sample_weights(self, y_train):
        positives = y_train.sum(axis=0)
        negatives = y_train.shape[0] - positives
        positive_weights = np.ones_like(positives, dtype=np.float32)
        mask = positives > 0
        positive_weights[mask] = np.clip(negatives[mask] / positives[mask], 1.0, 20.0)
        sample_weight = np.ones(y_train.shape[0], dtype=np.float32)
        for idx, row in enumerate(y_train):
            active_weights = positive_weights[row > 0]
            if active_weights.size > 0:
                sample_weight[idx] = float(np.max(active_weights))

        print("\nPositive label weights:")
        for name, weight in zip(CONSTRAINT_LABELS, positive_weights):
            print(f"  {name:<25}: {weight:.4f}")

        print(
            f"  sample weights             : "
            f"min={sample_weight.min():.4f}, max={sample_weight.max():.4f}, mean={sample_weight.mean():.4f}"
        )
        return sample_weight

    def run(self):
        print("=" * 70)
        print("CONSTRAINT CLASSIFIER — TRAINING PIPELINE")
        print("=" * 70)

        X, y = self.load_data()
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_data(X, y)
        self.build_model(X_train.shape[1], y_train.shape[1])
        self.train(X_train, y_train, X_val, y_val, self.label_sample_weights(y_train))
        self.evaluate(X_test, y_test)

        print("\n" + "=" * 70)
        print("DONE")
        print("=" * 70)
        print(f"  Model   → {config.CONSTRAINT_CLASSIFIER_PATH}")
        print(f"  Scaler  → {config.CONSTRAINT_SCALER_PATH}")

    def cross_validate(self, X, y, folds: int = 5):
        from sklearn.model_selection import StratifiedKFold, train_test_split

        strata = np.clip(y.sum(axis=1).astype(int), 0, 3)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_SEED)
        metrics = []
        fold = 0
        for train_idx, test_idx in skf.split(X, strata):
            fold += 1
            print(f"\n--- Fold {fold}/{folds} ---")
            X_rest = X[train_idx]
            y_rest = y[train_idx]

            X_test = X[test_idx]
            y_test = y[test_idx]

            X_tr, X_val, y_tr, y_val = train_test_split(
                X_rest,
                y_rest,
                test_size=config.VALIDATION_RATIO / (config.TRAIN_RATIO + config.VALIDATION_RATIO),
                random_state=config.RANDOM_SEED + fold,
                stratify=np.clip(y_rest.sum(axis=1).astype(int), 0, 3),
            )

            scaler = StandardScaler()
            scaler.fit(X_tr)
            joblib.dump(scaler, str(config.CONSTRAINT_SCALER_PATH).replace('.pkl', f'.fold{fold}.pkl'))
            X_tr = scaler.transform(X_tr)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            self.build_model(X_tr.shape[1], y_tr.shape[1])
            self.train(X_tr, y_tr, X_val, y_val, self.label_sample_weights(y_tr))
            res = self.evaluate(X_test, y_test)
            metrics.append(res)

        import numpy as _np
        mean = _np.mean(_np.array(metrics), axis=0)
        print("\nCross-validation summary (mean over folds):")
        for name, value in zip(self.model.metrics_names, mean):
            print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train constraint classifier")
    parser.add_argument("data", nargs="?", help="Path to constraint data json")
    parser.add_argument("--cv", type=int, default=0, help="Run k-fold cross-validation (k)")
    args = parser.parse_args()

    trainer = ConstraintClassifierTrainer(args.data)
    if args.cv and args.cv > 1:
        X, y = trainer.load_data()
        trainer.cross_validate(X, y, folds=args.cv)
    else:
        trainer.run()
