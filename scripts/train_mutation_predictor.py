"""
Training script for the Mutation Predictor model.

Usage:
    python scripts/train_mutation_predictor.py
    python scripts/train_mutation_predictor.py data/mutation_data.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import argparse
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    # ensure `src` package modules that use bare imports (e.g. `import config`) are importable
    src_path = PROJECT_ROOT / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

import src.config as config
from src.feature_extraction import ScheduleFeatureExtractor
from src.models import MutationPredictorModel


IMPACT_TO_CLASS = {
    "improve": 0,
    "neutral": 1,
    "worsen": 2,
}

MUTATION_TYPE_TO_ID = {
    "swap": 1,
    "move": 2,
    "shift": 3,
    "clear_day": 4,
}


class MutationPredictorTrainer:
    def __init__(self, data_path: str | None = None):
        self.data_path = Path(data_path) if data_path else PROJECT_ROOT / "data" / "mutation_data.json"
        self.extractor = ScheduleFeatureExtractor()
        self.scaler = StandardScaler()
        self.model = None
        self.history = None

        np.random.seed(config.RANDOM_SEED)
        tf.random.set_seed(config.RANDOM_SEED)

    def load_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Mutation data not found at {self.data_path}")

        raw = json.loads(self.data_path.read_text(encoding="utf-8-sig"))
        samples = raw.get("schedules", raw.get("data", raw)) if isinstance(raw, dict) else raw
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"No samples found in {self.data_path}")

        def extract_week_schedule(sched):
            """Extract week_schedule from either nested dict or direct array format."""
            if isinstance(sched, dict):
                return sched.get("week_schedule")
            elif isinstance(sched, list):
                return sched
            return None

        X = []
        y = []
        class_ids = []

        for sample in samples:
            orig_sched = sample.get("original_schedule") or sample.get("current_schedule")
            ws = extract_week_schedule(orig_sched)
            if ws is None:
                continue

            impact = str(sample.get("impact", "neutral")).lower()
            if impact not in IMPACT_TO_CLASS:
                continue

            # Extract features from schedule (extractor expects dict with week_schedule key)
            schedule_features = self.extractor.extract_features({"week_schedule": ws})

            mutation_info = sample.get("mutation_info", {})
            mutation_type = str(mutation_info.get("type", "")).lower()
            mutation_type_id = MUTATION_TYPE_TO_ID.get(mutation_type, 0)
            position = float(mutation_info.get("position", 0))
            current_fitness = float(sample.get("original_fitness", sample.get("fitness", 0.0)))

            combined = np.concatenate(
                [schedule_features, np.array([current_fitness, mutation_type_id, position], dtype=np.float32)]
            )

            expected_size = config.MUTATION_PREDICTOR_CONFIG["input_dim"]
            if combined.shape[0] < expected_size:
                combined = np.pad(combined, (0, expected_size - combined.shape[0]))
            else:
                combined = combined[:expected_size]

            class_id = IMPACT_TO_CLASS[impact]
            X.append(combined.astype(np.float32))
            y.append(class_id)
            class_ids.append(class_id)

        if not X:
            raise ValueError("No valid mutation samples could be parsed.")

        X_arr = np.array(X, dtype=np.float32)
        class_ids_arr = np.array(class_ids, dtype=np.int32)
        y_arr = to_categorical(np.array(y, dtype=np.int32), num_classes=3)

        counts = np.bincount(class_ids_arr, minlength=3)
        print("\nClass distribution:")
        for name, idx in IMPACT_TO_CLASS.items():
            print(f"  {name:<8}: {counts[idx]}")

        return X_arr, y_arr, class_ids_arr

    def preprocess_data(self, X, y, class_ids):
        X_train, X_tmp, y_train, y_tmp, cls_train, cls_tmp = train_test_split(
            X,
            y,
            class_ids,
            test_size=1.0 - config.TRAIN_RATIO,
            random_state=config.RANDOM_SEED,
            stratify=class_ids,
        )
        test_fraction = config.TEST_RATIO / (config.TEST_RATIO + config.VALIDATION_RATIO)
        X_val, X_test, y_val, y_test, cls_val, cls_test = train_test_split(
            X_tmp,
            y_tmp,
            cls_tmp,
            test_size=test_fraction,
            random_state=config.RANDOM_SEED,
            stratify=cls_tmp,
        )

        self.scaler.fit(X_train)
        joblib.dump(self.scaler, config.MUTATION_SCALER_PATH)
        X_train = self.scaler.transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        return X_train, X_val, X_test, y_train, y_val, y_test, cls_train, cls_test

    def build_model(self, input_dim: int):
        self.model = MutationPredictorModel.build(input_dim=input_dim, num_classes=3)
        self.model.summary()

    def train(self, X_train, y_train, X_val, y_val, class_weight):
        cfg = config.MUTATION_PREDICTOR_CONFIG
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg.get("early_stopping_patience", 20),
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=6,
                min_lr=1e-7,
                verbose=1,
            ),
            keras.callbacks.CSVLogger(str(config.LOGS_DIR / "mutation_predictor_training.csv")),
            keras.callbacks.ModelCheckpoint(
                str(config.MUTATION_PREDICTOR_PATH),
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
            class_weight=class_weight,
            verbose=1,
        )

    def evaluate(self, X_test, y_test):
        results = self.model.evaluate(X_test, y_test, verbose=1)
        print("\nTest metrics:")
        for name, value in zip(self.model.metrics_names, results):
            print(f"  {name}: {value:.6f}")
        y_true = np.argmax(y_test, axis=1)
        y_pred = np.argmax(self.model.predict(X_test, verbose=0), axis=1)
        names = [name for name, _ in sorted(IMPACT_TO_CLASS.items(), key=lambda item: item[1])]
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, target_names=names, zero_division=0))
        print("Confusion matrix:")
        print(confusion_matrix(y_true, y_pred))
        return results

    def class_weight(self, class_ids):
        counts = np.bincount(class_ids, minlength=3).astype(np.float32)
        total = float(np.sum(counts))
        weights = {
            idx: total / (len(counts) * count)
            for idx, count in enumerate(counts)
            if count > 0
        }
        print("\nClass weights:")
        for idx, weight in weights.items():
            print(f"  class {idx}: {weight:.4f}")
        return weights

    def run(self):
        print("=" * 70)
        print("MUTATION PREDICTOR — TRAINING PIPELINE")
        print("=" * 70)

        X, y, class_ids = self.load_data()
        self.build_model(config.MUTATION_PREDICTOR_CONFIG["input_dim"])

        # default single-run training
        X_train, X_val, X_test, y_train, y_val, y_test, cls_train, _ = self.preprocess_data(X, y, class_ids)
        self.train(X_train, y_train, X_val, y_val, self.class_weight(cls_train))
        self.evaluate(X_test, y_test)

        print("\n" + "=" * 70)
        print("DONE")
        print("=" * 70)
        print(f"  Model → {config.MUTATION_PREDICTOR_PATH}")
        print(f"  Scaler → {config.MUTATION_SCALER_PATH}")

    def cross_validate(self, X, y, class_ids, folds: int = 5):
        from sklearn.model_selection import StratifiedKFold, train_test_split

        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_SEED)
        fold_metrics = []
        fold_idx = 0
        for train_val_idx, test_idx in skf.split(X, class_ids):
            fold_idx += 1
            print(f"\n--- Fold {fold_idx}/{folds} ---")
            X_rest = X[train_val_idx]
            y_rest = y[train_val_idx]
            cls_rest = class_ids[train_val_idx]

            test_X = X[test_idx]
            test_y = y[test_idx]

            # split rest into train/val
            X_tr, X_val, y_tr, y_val, cls_tr, cls_val = train_test_split(
                X_rest,
                y_rest,
                cls_rest,
                test_size=config.VALIDATION_RATIO / (config.TRAIN_RATIO + config.VALIDATION_RATIO),
                random_state=config.RANDOM_SEED + fold_idx,
                stratify=cls_rest,
            )

            # fit scaler per-fold
            scaler_obj = StandardScaler()
            scaler_obj.fit(X_tr)
            joblib.dump(scaler_obj, str(config.MUTATION_SCALER_PATH).replace('.pkl', f'.fold{fold_idx}.pkl'))
            X_tr = scaler_obj.transform(X_tr)
            X_val = scaler_obj.transform(X_val)
            X_test = scaler_obj.transform(test_X)

            # build & train fresh model per-fold
            self.build_model(X_tr.shape[1])
            self.train(X_tr, y_tr, X_val, y_val, self.class_weight(cls_tr))
            res = self.evaluate(X_test, test_y)
            fold_metrics.append(res)

        # summarize
        import numpy as _np
        arr = _np.array(fold_metrics)
        mean = arr.mean(axis=0)
        print("\nCross-validation summary (mean over folds):")
        for name, value in zip(self.model.metrics_names, mean):
            print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train mutation predictor")
    parser.add_argument("data", nargs="?", help="Path to mutation data json")
    parser.add_argument("--cv", type=int, default=0, help="Run k-fold cross-validation (k)")
    args = parser.parse_args()

    trainer = MutationPredictorTrainer(args.data)
    if args.cv and args.cv > 1:
        X, y, class_ids = trainer.load_data()
        trainer.cross_validate(X, y, class_ids, folds=args.cv)
    else:
        trainer.run()
