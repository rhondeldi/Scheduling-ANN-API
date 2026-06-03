"""
Evaluate the trained ANN models on held-out splits and print a concise report.

Usage:
    python scripts/evaluate_models_report.py
"""
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
from tensorflow import keras

# Allow running this file directly via "python scripts/evaluate_models_report.py".
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config
from src.feature_extraction import FitnessFeatureExtractor


RNG = np.random.default_rng(config.RANDOM_SEED)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def normalize_samples(data):
    if isinstance(data, dict):
        for key in (
            "schedules",
            "data",
            "samples",
        ):
            value = data.get(key)
            if isinstance(value, list):
                return value
        return []
    if isinstance(data, list):
        return data
    return []


def train_test_split_indices(n: int, train_ratio: float, validation_ratio: float, test_ratio: float):
    indices = np.arange(n)
    RNG.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * validation_ratio)

    if val_end >= n:
        val_end = n - max(1, int(n * test_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    if len(test_idx) == 0:
        test_idx = indices[-max(1, int(n * test_ratio)) :]

    return train_idx, val_idx, test_idx


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    return float(math.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - (ss_res / ss_tot if ss_tot != 0 else 0.0)


def micro_f1(y_true, y_pred):
    y_true = y_true.astype(int).flatten()
    y_pred = y_pred.astype(int).flatten()
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


def load_model(path: Path):
    return keras.models.load_model(path, compile=False)


def evaluate_fitness():
    print("=== FITNESS ===")
    samples = normalize_samples(load_json(PROJECT_ROOT / "data" / "manual_schedules_with_fitness.json"))
    extractor = FitnessFeatureExtractor()

    X, y = [], []
    for sample in samples:
        ws = sample.get("schedule", {}).get("week_schedule") if isinstance(sample.get("schedule"), dict) else sample.get("week_schedule")
        if ws is None:
            continue
        X.append(extractor.extract({"week_schedule": ws}))
        y.append(float(sample["fitness"]))

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

    train_idx, _, test_idx = train_test_split_indices(len(X), config.TRAIN_RATIO, config.VALIDATION_RATIO, config.TEST_RATIO)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    feature_scaler = joblib.load(config.FEATURE_SCALER_PATH)
    fitness_scaler = joblib.load(config.FITNESS_SCALER_PATH)

    X_test_s = feature_scaler.transform(X_test)
    y_test_s = fitness_scaler.transform(y_test)

    model = load_model(config.FITNESS_PREDICTOR_PATH)
    y_pred_s = model.predict(X_test_s, verbose=0)
    y_pred = fitness_scaler.inverse_transform(y_pred_s)

    print({
        "samples": int(len(X)),
        "test_samples": int(len(test_idx)),
        "scaled_mse": mse(y_test_s, y_pred_s),
        "scaled_rmse": rmse(y_test_s, y_pred_s),
        "r2": r2(y_test, y_pred),
        "mae": mae(y_test, y_pred),
        "fitness_range": [float(np.min(y)), float(np.max(y))],
    })


def main():
    evaluate_fitness()


if __name__ == "__main__":
    main()
