"""
Training script for the Crossover Recommender model.

Usage:
    python scripts/train_crossover_recommender.py
    python scripts/train_crossover_recommender.py data/crossover_data.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import top_k_accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
import argparse
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    # ensure `src` package modules that use bare imports (e.g. `import config`) are importable
    src_path = PROJECT_ROOT / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

import src.config as config
from src.models import CrossoverRecommenderModel


class CrossoverRecommenderTrainer:
    def __init__(self, data_path: str | None = None):
        self.data_path = Path(data_path) if data_path else PROJECT_ROOT / "data" / "crossover_data.json"
        self.model = None
        self.history = None

        np.random.seed(config.RANDOM_SEED)
        tf.random.set_seed(config.RANDOM_SEED)

    def load_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Crossover data not found at {self.data_path}")

        raw = json.loads(self.data_path.read_text(encoding="utf-8-sig"))
        samples = raw.get("schedules", raw.get("data", raw)) if isinstance(raw, dict) else raw
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"No samples found in {self.data_path}")

        def normalize_week_schedule(item):
            """Normalize various week_schedule representations into an (N,3) float32 array.

            Accepts:
            - nested 6x24x3 lists (rows of slots)
            - flat list of numbers (multiple of 3)
            - list of triplet lists or list of dicts (will extract numeric values)
            Returns None if it cannot produce a valid (N,3) array.
            """
            # unwrap common dict wrappers
            if isinstance(item, dict):
                for key in ("week_schedule", "weekSchedule", "schedule", "week"):
                    if key in item:
                        item = item[key]
                        break
                else:
                    # try to find a list-valued field
                    for v in item.values():
                        if isinstance(v, list):
                            item = v
                            break

            def gather_triplets(obj):
                """Yield triplet-like lists found anywhere inside obj."""
                if obj is None:
                    return
                if isinstance(obj, (list, tuple)):
                    # flat numeric list -> split into triplets
                    if all(isinstance(x, (int, float)) for x in obj):
                        arr = list(obj)
                        if len(arr) % 3 != 0:
                            return
                        for i in range(0, len(arr), 3):
                            yield [arr[i], arr[i + 1], arr[i + 2]]
                        return

                    # list of lists/tuples/dicts: recurse
                    for elem in obj:
                        if isinstance(elem, (list, tuple)):
                            # if elem itself looks like a triplet of numbers
                            if len(elem) == 3 and all(isinstance(x, (int, float)) for x in elem):
                                yield [float(elem[0]), float(elem[1]), float(elem[2])]
                            else:
                                for t in gather_triplets(elem):
                                    yield t
                        elif isinstance(elem, dict):
                            # try numeric keys or named keys
                            if all(k in elem for k in ("day", "slot", "subject")):
                                yield [float(elem.get("day", 0)), float(elem.get("slot", 0)), float(elem.get("subject", 0))]
                            else:
                                # try values order
                                vals = list(elem.values())
                                if len(vals) >= 3 and all(isinstance(x, (int, float)) for x in vals[:3]):
                                    yield [float(vals[0]), float(vals[1]), float(vals[2])]
                                else:
                                    for t in gather_triplets(vals):
                                        yield t
                        else:
                            # non-list scalar – can't form triplet here
                            continue
                elif isinstance(obj, dict):
                    for v in obj.values():
                        for t in gather_triplets(v):
                            yield t

            triplets = list(gather_triplets(item))
            if not triplets:
                return None
            try:
                arr = np.asarray(triplets, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[1] != 3:
                    return None
                # enforce canonical length (144 slots = 6 days * 24 slots)
                TARGET_LEN = 144
                if arr.shape[0] < TARGET_LEN:
                    pad_rows = TARGET_LEN - arr.shape[0]
                    pad = np.zeros((pad_rows, 3), dtype=np.float32)
                    arr = np.vstack([arr, pad])
                elif arr.shape[0] > TARGET_LEN:
                    arr = arr[:TARGET_LEN]
                return arr
            except Exception:
                return None

        parent1 = []
        parent2 = []
        labels = []

        for idx, sample in enumerate(samples):
            p1_item = sample.get("parent1")
            p2_item = sample.get("parent2")
            try:
                p1 = normalize_week_schedule(p1_item)
                p2 = normalize_week_schedule(p2_item)
            except Exception as exc:
                print(f"[warning] sample {idx}: exception normalizing parents: {exc}")
                continue

            if p1 is None or p2 is None:
                # skip malformed entries but warn with sample index for debugging
                print(f"[warning] sample {idx}: could not parse parent schedules (p1={type(p1_item)}, p2={type(p2_item)})")
                continue

            point = sample.get("crossover_point")
            if point is None:
                point = sample.get("crossover_bin")
            if point is None:
                print(f"[warning] sample {idx}: missing crossover point")
                continue

            try:
                point = int(point)
            except Exception:
                print(f"[warning] sample {idx}: invalid crossover point value {point}")
                continue
            point = max(0, min(143, point))

            parent1.append(p1)
            parent2.append(p2)
            labels.append(point)

        if not parent1:
            raise ValueError("No valid crossover samples could be parsed.")

        X1 = np.array(parent1, dtype=np.float32)
        X2 = np.array(parent2, dtype=np.float32)
        class_ids = np.array(labels, dtype=np.int32)
        y = self.soft_crossover_targets(class_ids)

        counts = Counter(class_ids.tolist())
        print("\nCrossover point distribution:")
        print(f"  unique points : {len(counts)}")
        print(f"  min / max     : {min(counts)} / {max(counts)}")
        print("  top points    : " + ", ".join(f"{point}:{count}" for point, count in counts.most_common(8)))

        return X1, X2, y, class_ids

    def soft_crossover_targets(self, class_ids, sigma: float = 2.0):
        positions = np.arange(144, dtype=np.float32)
        targets = []
        for point in class_ids:
            distances = positions - float(point)
            probs = np.exp(-(distances ** 2) / (2.0 * sigma ** 2))
            probs /= probs.sum()
            targets.append(probs)
        return np.array(targets, dtype=np.float32)

    def preprocess_data(self, X1, X2, y, class_ids):
        strata = np.clip(class_ids // 12, 0, 11)
        stratify = strata if min(Counter(strata.tolist()).values()) >= 2 else None

        X1_train, X1_tmp, X2_train, X2_tmp, y_train, y_tmp, cls_train, cls_tmp = train_test_split(
            X1,
            X2,
            y,
            class_ids,
            test_size=1.0 - config.TRAIN_RATIO,
            random_state=config.RANDOM_SEED,
            stratify=stratify,
        )
        test_fraction = config.TEST_RATIO / (config.TEST_RATIO + config.VALIDATION_RATIO)
        tmp_strata = np.clip(cls_tmp // 12, 0, 11)
        tmp_stratify = tmp_strata if min(Counter(tmp_strata.tolist()).values()) >= 2 else None
        X1_val, X1_test, X2_val, X2_test, y_val, y_test, cls_val, cls_test = train_test_split(
            X1_tmp,
            X2_tmp,
            y_tmp,
            cls_tmp,
            test_size=test_fraction,
            random_state=config.RANDOM_SEED,
            stratify=tmp_stratify,
        )
        return X1_train, X1_val, X1_test, X2_train, X2_val, X2_test, y_train, y_val, y_test, cls_train, cls_test

    def build_model(self):
        self.model = CrossoverRecommenderModel.build()
        self.model.summary()

    def train(self, X1_train, X2_train, y_train, X1_val, X2_val, y_val, sample_weight):
        cfg = config.CROSSOVER_RECOMMENDER_CONFIG
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=20,
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
            keras.callbacks.CSVLogger(str(config.LOGS_DIR / "crossover_recommender_training.csv")),
            keras.callbacks.ModelCheckpoint(
                str(config.CROSSOVER_RECOMMENDER_PATH),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
        ]

        self.history = self.model.fit(
            [X1_train, X2_train],
            y_train,
            validation_data=([X1_val, X2_val], y_val),
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            callbacks=callbacks,
            sample_weight=sample_weight,
            verbose=1,
        )

    def evaluate(self, X1_test, X2_test, y_test, cls_test):
        results = self.model.evaluate([X1_test, X2_test], y_test, verbose=1)
        print("\nTest metrics:")
        for name, value in zip(self.model.metrics_names, results):
            print(f"  {name}: {value:.6f}")
        probs = self.model.predict([X1_test, X2_test], verbose=0)
        pred = np.argmax(probs, axis=1)
        for tolerance in (0, 3, 6, 12):
            within = np.mean(np.abs(pred - cls_test) <= tolerance)
            print(f"  within_{tolerance:02d}_slots: {within:.6f}")
        for k in (3, 5, 10):
            print(f"  top_{k}_exact: {top_k_accuracy_score(cls_test, probs, k=k, labels=np.arange(144)):.6f}")
        return results

    def point_sample_weights(self, class_ids):
        counts = Counter(class_ids.tolist())
        median_count = float(np.median(list(counts.values())))
        weights = np.array(
            [np.clip(median_count / counts[int(point)], 0.25, 4.0) for point in class_ids],
            dtype=np.float32,
        )
        print(f"\nSample weights: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")
        return weights

    def run(self):
        print("=" * 70)
        print("CROSSOVER RECOMMENDER — TRAINING PIPELINE")
        print("=" * 70)

        X1, X2, y, class_ids = self.load_data()
        X1_train, X1_val, X1_test, X2_train, X2_val, X2_test, y_train, y_val, y_test, cls_train, cls_test = self.preprocess_data(X1, X2, y, class_ids)
        self.build_model()
        self.train(X1_train, X2_train, y_train, X1_val, X2_val, y_val, self.point_sample_weights(cls_train))
        self.evaluate(X1_test, X2_test, y_test, cls_test)

        print("\n" + "=" * 70)
        print("DONE")
        print("=" * 70)
        print(f"  Model → {config.CROSSOVER_RECOMMENDER_PATH}")

    def cross_validate(self, X1, X2, y, class_ids, folds: int = 5):
        from sklearn.model_selection import StratifiedKFold

        strata = np.clip(class_ids // 12, 0, 11)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_SEED)
        fold = 0
        metrics = []
        for train_val_idx, test_idx in skf.split(X1, strata):
            fold += 1
            print(f"\n--- Fold {fold}/{folds} ---")
            X1_rest = X1[train_val_idx]
            X2_rest = X2[train_val_idx]
            y_rest = y[train_val_idx]
            cls_rest = class_ids[train_val_idx]

            X1_test = X1[test_idx]
            X2_test = X2[test_idx]
            y_test = y[test_idx]
            cls_test = class_ids[test_idx]

            # split rest into train/val
            X1_tr, X1_val, X2_tr, X2_val, y_tr, y_val, cls_tr, cls_val = train_test_split(
                X1_rest, X2_rest, y_rest, cls_rest,
                test_size=config.VALIDATION_RATIO / (config.TRAIN_RATIO + config.VALIDATION_RATIO),
                random_state=config.RANDOM_SEED + fold,
                stratify=np.clip(cls_rest // 12, 0, 11),
            )

            # build & train fresh model per-fold
            self.build_model()
            self.train(X1_tr, X2_tr, y_tr, X1_val, X2_val, y_val, self.point_sample_weights(cls_tr))
            res = self.evaluate(X1_test, X2_test, y_test, cls_test)
            metrics.append(res)

        import numpy as _np
        mean = _np.mean(_np.array(metrics), axis=0)
        print("\nCross-validation summary (mean over folds):")
        for name, value in zip(self.model.metrics_names, mean):
            print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train crossover recommender")
    parser.add_argument("data", nargs="?", help="Path to crossover data json")
    parser.add_argument("--cv", type=int, default=0, help="Run k-fold cross-validation (k)")
    args = parser.parse_args()

    trainer = CrossoverRecommenderTrainer(args.data)
    if args.cv and args.cv > 1:
        X1, X2, y, class_ids = trainer.load_data()
        trainer.cross_validate(X1, X2, y, class_ids, folds=args.cv)
    else:
        trainer.run()
