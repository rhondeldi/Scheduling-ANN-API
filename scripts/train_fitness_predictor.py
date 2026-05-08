"""
Training script for the Fitness Predictor Model.

Usage:
    python scripts/train_fitness_predictor.py
    python scripts/train_fitness_predictor.py data/training_data.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")          # headless — saves PNG without needing a display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

# ── path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config
from src.models import FitnessPredictorModel
from src.feature_extraction import FitnessFeatureExtractor


# ══════════════════════════════════════════════════════════════════════════════
class FitnessPredictorTrainer:
    """End-to-end trainer for the schedule fitness predictor ANN."""

    def __init__(self, data_path: str | None = None):
        self.data_path        = Path(data_path) if data_path else PROJECT_ROOT / "data" / "training_data.json"
        self.feature_extractor = FitnessFeatureExtractor()
        self.scaler_X          = StandardScaler()
        self.scaler_y          = StandardScaler()
        self.model             = None
        self.history           = None

        np.random.seed(config.RANDOM_SEED)
        tf.random.set_seed(config.RANDOM_SEED)

    # ── data loading ───────────────────────────────────────────────────────────

    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        print(f"Loading data from: {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Training data not found at {self.data_path}.\n"
                "Run generate_synthetic_variants.py first:\n"
                "  python scripts/generate_synthetic_variants.py "
                "--input data/manual_schedules_with_fitness.json "
                "--output data/training_data.json --target 5000"
            )

        with open(self.data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # normalise top-level format
        if isinstance(raw, dict):
            samples = raw.get("schedules", raw.get("data", []))
        else:
            samples = raw

        if not isinstance(samples, list) or len(samples) == 0:
            raise ValueError(f"No samples found in {self.data_path}")

        X, y = [], []
        skipped = 0

        for sample in samples:
            fitness = sample.get("fitness")
            if not isinstance(fitness, (int, float)):
                skipped += 1
                continue

            # skip empty-schedule sentinel
            if float(fitness) <= -24.0:
                skipped += 1
                continue

            # locate week_schedule
            if isinstance(sample.get("schedule"), dict):
                payload = sample["schedule"]
            elif "week_schedule" in sample:
                payload = {"week_schedule": sample["week_schedule"]}
            else:
                skipped += 1
                continue

            features = self.feature_extractor.extract(payload)
            X.append(features)
            y.append(float(fitness))

        if len(X) == 0:
            raise ValueError("All samples were skipped — check training data format.")

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)

        print(f"  Loaded   : {len(X)} samples  (skipped {skipped})")
        print(f"  Features : {X.shape[1]}")
        print(f"  Fitness  : min={y.min():.4f}  max={y.max():.4f}  mean={y.mean():.4f}")

        return X, y

    # ── preprocessing ──────────────────────────────────────────────────────────

    def preprocess_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        print("\nPreprocessing data...")

        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y,
            test_size=1.0 - config.TRAIN_RATIO,
            random_state=config.RANDOM_SEED,
        )
        test_fraction = config.TEST_RATIO / (config.TEST_RATIO + config.VALIDATION_RATIO)
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp,
            test_size=test_fraction,
            random_state=config.RANDOM_SEED,
        )

        X_train = self.scaler_X.fit_transform(X_train)
        X_val   = self.scaler_X.transform(X_val)
        X_test  = self.scaler_X.transform(X_test)

        # scale target for stable optimization and restore via inverse_transform
        y_train = self.scaler_y.fit_transform(y_train)
        y_val   = self.scaler_y.transform(y_val)
        y_test  = self.scaler_y.transform(y_test)
        print(f"  Train y range : {y_train.min():.3f} – {y_train.max():.3f}")

        print(f"  Train : {X_train.shape[0]} samples")
        print(f"  Val   : {X_val.shape[0]}   samples")
        print(f"  Test  : {X_test.shape[0]}   samples")

        joblib.dump(self.scaler_X, config.FEATURE_SCALER_PATH)
        joblib.dump(self.scaler_y, config.FITNESS_SCALER_PATH)
        print(f"  Scaler saved → {config.MODELS_DIR}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    # ── model ──────────────────────────────────────────────────────────────────

    def build_model(self, input_dim: int):
        print("\nBuilding model...")
        self.model = FitnessPredictorModel.build(input_dim)
        self.model.summary()

    # ── training ───────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
    ):
        print("\nStarting training...")
        cfg = config.FITNESS_PREDICTOR_CONFIG

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
                patience=10,
                min_lr=1e-7,
                verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                str(config.FITNESS_PREDICTOR_PATH),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            keras.callbacks.TensorBoard(
                log_dir=str(config.LOGS_DIR / "fitness_predictor"),
                histogram_freq=1,
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

        print("\nTraining complete.")

    # ── evaluation ─────────────────────────────────────────────────────────────

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        print("\nEvaluating on test set...")

        results = self.model.evaluate(X_test, y_test, verbose=1)
        print("\nTest metrics:")
        for name, val in zip(self.model.metrics_names, results):
            print(f"  {name}: {val:.6f}")

        y_pred = self.model.predict(X_test, verbose=0)

        y_test_orig = self.scaler_y.inverse_transform(y_test)
        y_pred_orig = self.scaler_y.inverse_transform(y_pred)

        r2  = r2_score(y_test_orig, y_pred_orig)
        mae = mean_absolute_error(y_test_orig, y_pred_orig)

        # SMAPE — robust to near-zero targets
        num   = np.abs(y_test_orig - y_pred_orig)
        denom = (np.abs(y_test_orig) + np.abs(y_pred_orig)) / 2.0
        smape = float(100.0 * np.mean(num / (denom + 1e-10)))

        print(f"\nOriginal-scale metrics:")
        print(f"  R²    : {r2:.4f}")
        print(f"  MAE   : {mae:.4f}")
        print(f"  SMAPE : {smape:.2f}%")

        # scatter: predicted vs actual
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(y_test_orig, y_pred_orig, alpha=0.4, s=12)
        lims = [
            min(y_test_orig.min(), y_pred_orig.min()) - 1,
            max(y_test_orig.max(), y_pred_orig.max()) + 1,
        ]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel("Actual fitness")
        ax.set_ylabel("Predicted fitness")
        ax.set_title(f"Predicted vs Actual  (R²={r2:.3f}, MAE={mae:.3f})")
        ax.grid(True, alpha=0.3)
        scatter_path = config.LOGS_DIR / "fitness_predictor_scatter.png"
        fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Scatter plot → {scatter_path}")

        return results

    # ── plot training curves ───────────────────────────────────────────────────

    def plot_training_history(self):
        if self.history is None:
            print("No training history available.")
            return

        h = self.history.history
        epochs = range(1, len(h["loss"]) + 1)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Fitness Predictor — Training History", fontsize=14)

        def _plot(ax, train_key, val_key, title, ylabel):
            if train_key in h:
                ax.plot(epochs, h[train_key], label="Train")
            if val_key in h:
                ax.plot(epochs, h[val_key],   label="Val", linestyle="--")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True, alpha=0.3)

        _plot(axes[0, 0], "loss",  "val_loss",  "Loss (MSE)",  "MSE")
        _plot(axes[0, 1], "mae",   "val_mae",   "MAE",         "MAE")
        _plot(axes[1, 0], "mse",   "val_mse",   "MSE",         "MSE")
        _plot(axes[1, 1], "rmse",  "val_rmse",  "RMSE",        "RMSE")

        plt.tight_layout()
        plot_path = config.LOGS_DIR / "fitness_predictor_training.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nTraining curves → {plot_path}")

    # ── full pipeline ──────────────────────────────────────────────────────────

    def run(self):
        print("=" * 70)
        print("FITNESS PREDICTOR — TRAINING PIPELINE")
        print("=" * 70)

        X, y = self.load_data()
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_data(X, y)
        self.build_model(X_train.shape[1])
        self.train(X_train, y_train, X_val, y_val)
        self.evaluate(X_test, y_test)
        self.plot_training_history()

        print("\n" + "=" * 70)
        print("DONE")
        print("=" * 70)
        print(f"  Model   → {config.FITNESS_PREDICTOR_PATH}")
        print(f"  Scalers → {config.MODELS_DIR}")
        print(f"  Logs    → {config.LOGS_DIR}")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    trainer = FitnessPredictorTrainer(data_path)
    trainer.run()