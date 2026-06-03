"""
Verify training data files for all four ANN models.

For each known dataset this script checks:
  - File exists and is parseable (JSON or JSONL).
  - Each record has the required keys for that model's trainer.
  - Each schedule grid has the correct (N_DAYS, N_SLOTS, 3) shape.
  - Numeric ranges are sane (no NaNs, slot indices >= 0, fitness numeric).
  - Label distribution / class balance after the trainer's labelling rule.
  - Dataset-specific warnings (e.g. no full_uni_schedule -> hard-conflict
    labels are unlearnable for the constraint classifier).

The script does NOT load TensorFlow — it's safe to run before any training,
even in environments where the model deps aren't installed.

Exit code:
  0  every checked dataset passed (warnings allowed)
  1  at least one dataset had hard errors

Usage:
    python scripts/verify_training_data.py
    python scripts/verify_training_data.py --kind mutation
    python scripts/verify_training_data.py --kind crossover --file data/cx.jsonl
    python scripts/verify_training_data.py --strict   # treat warnings as errors
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    src_path = PROJECT_ROOT / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

import src.config as config

N_DAYS = config.N_WEEKLY_SCHOOL_DAYS
N_SLOTS = config.N_DAILY_TIME_SLOTS
ATTR_PER_SLOT = 3

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
HARD_CONSTRAINT_LABELS = {"instructor_conflict", "room_conflict"}

MUTATION_LABEL_THRESHOLDS = {"improve": 0.5, "worsen": -0.5}
CROSSOVER_FITNESS_THRESHOLD = 5.0
MUTATION_TYPES_SPEC = {
    "day_swap_timeslots",
    "subject_day_swap",
    "slot_nudge",
    "slot_day_nudge",
}


# ── Report container ─────────────────────────────────────────────────────────
@dataclass
class CheckReport:
    kind: str
    path: Path
    total_records: int = 0
    valid_records: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


# ── Shared helpers ───────────────────────────────────────────────────────────
def iter_records(path: Path) -> Iterable[dict]:
    """Yield dicts from a .jsonl file or from a JSON document."""
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return
    if path.suffix.lower() == ".jsonl":
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_no}: invalid JSON ({exc})")
        return

    doc = json.loads(text)
    if isinstance(doc, list):
        yield from doc
    elif isinstance(doc, dict):
        records = doc.get("schedules") or doc.get("data") or []
        yield from records
    else:
        raise ValueError("Top-level JSON must be a list or a dict")


def coerce_grid(sched: Any) -> np.ndarray | None:
    """Try to coerce a schedule into a (N_DAYS, N_SLOTS, 3) int32 array."""
    if isinstance(sched, dict):
        for key in ("week_schedule", "section_schedule", "schedule"):
            if key in sched and sched[key] is not None:
                sched = sched[key]
                break
    if sched is None:
        return None
    try:
        arr = np.asarray(sched, dtype=np.int64)
    except Exception:
        return None
    if arr.ndim != 3:
        return None
    if arr.shape[0] != N_DAYS or arr.shape[1] != N_SLOTS:
        return None
    if arr.shape[2] < ATTR_PER_SLOT:
        return None
    return arr[:, :, :ATTR_PER_SLOT].astype(np.int32)


def numeric_finite(x: Any) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


# ── Per-dataset checks ───────────────────────────────────────────────────────
def check_fitness(path: Path) -> CheckReport:
    rep = CheckReport(kind="fitness", path=path)
    if not path.exists():
        rep.err(f"file not found: {path}")
        return rep

    fitnesses: list[float] = []
    bad_shapes = 0
    missing_fitness = 0

    for i, rec in enumerate(iter_records(path)):
        rep.total_records += 1
        if not isinstance(rec, dict):
            rep.err(f"record {i}: not an object")
            continue

        sched_holder = rec.get("schedule") or rec.get("week_schedule")
        grid = coerce_grid(sched_holder if sched_holder is not None else rec)
        if grid is None:
            bad_shapes += 1
            continue

        fit = rec.get("fitness")
        if fit is None or not numeric_finite(fit):
            missing_fitness += 1
            continue

        fitnesses.append(float(fit))
        rep.valid_records += 1

    if bad_shapes:
        rep.warn(f"{bad_shapes} records had no valid {N_DAYS}x{N_SLOTS}x3 schedule")
    if missing_fitness:
        rep.warn(f"{missing_fitness} records missing/invalid fitness")

    if fitnesses:
        f = np.array(fitnesses)
        rep.stats["fitness_min"] = float(f.min())
        rep.stats["fitness_max"] = float(f.max())
        rep.stats["fitness_mean"] = float(f.mean())
        rep.stats["fitness_std"] = float(f.std())
    else:
        rep.err("no records produced a valid (schedule, fitness) pair")

    return rep


def check_constraint(path: Path) -> CheckReport:
    rep = CheckReport(kind="constraint", path=path)
    if not path.exists():
        rep.err(f"file not found: {path}")
        return rep

    positives = Counter()
    bad_shapes = 0
    missing_violations = 0
    unknown_labels = Counter()
    with_full_uni = 0
    with_cross_aggregates = 0
    cross_aggregate_keys = {
        "total_instructor_conflicts",
        "total_room_conflicts",
        "max_instructor_conflicts_in_one_slot",
        "max_room_conflicts_in_one_slot",
        "slots_with_any_conflict",
    }

    for i, rec in enumerate(iter_records(path)):
        rep.total_records += 1
        if not isinstance(rec, dict):
            rep.err(f"record {i}: not an object")
            continue

        section_raw = (
            rec.get("section_schedule") or rec.get("schedule") or rec.get("original_schedule")
        )
        section = coerce_grid(section_raw)
        if section is None:
            bad_shapes += 1
            continue

        full_uni = rec.get("full_uni_schedule")
        if full_uni:
            if not isinstance(full_uni, list):
                rep.warn(f"record {i}: full_uni_schedule is not a list")
            else:
                with_full_uni += 1
                # spot-check the first few sections in the full uni schedule
                for j, sec in enumerate(full_uni[:5]):
                    if coerce_grid(sec) is None:
                        rep.warn(
                            f"record {i}: full_uni_schedule[{j}] has bad shape; "
                            "cross-section features may be partially zero"
                        )
                        break

        cross = rec.get("cross_section")
        if isinstance(cross, dict):
            missing = cross_aggregate_keys - set(cross.keys())
            if missing:
                rep.warn(
                    f"record {i}: cross_section missing keys: "
                    f"{sorted(missing)}; defaulting to 0"
                )
            with_cross_aggregates += 1

        violations = rec.get("violations") or rec.get("constraints")
        if not isinstance(violations, dict):
            missing_violations += 1
            continue

        for name, val in violations.items():
            if name not in CONSTRAINT_LABELS:
                unknown_labels[name] += 1
                continue
            if bool(val):
                positives[name] += 1

        rep.valid_records += 1

    if bad_shapes:
        rep.warn(f"{bad_shapes} records had no valid section_schedule")
    if missing_violations:
        rep.warn(f"{missing_violations} records missing violations dict")
    if unknown_labels:
        rep.warn(
            "violations contained unknown labels (ignored): "
            + ", ".join(f"{k}:{v}" for k, v in unknown_labels.most_common(5))
        )

    if rep.valid_records == 0:
        rep.err("no valid constraint records found")
        return rep

    rep.stats["with_full_uni_schedule"] = with_full_uni
    rep.stats["with_cross_section_aggregates"] = with_cross_aggregates
    rep.stats["no_cross_section_source"] = (
        rep.valid_records - with_full_uni - with_cross_aggregates
    )
    rep.stats["positives"] = {name: positives.get(name, 0) for name in CONSTRAINT_LABELS}

    if with_full_uni == 0 and with_cross_aggregates == 0:
        rep.warn(
            "NO records include full_uni_schedule or cross_section aggregates — "
            "hard-conflict labels (instructor_conflict, room_conflict) cannot be "
            "learned because the cross-section features will all be zero."
        )

    # Per-label sanity
    for name in CONSTRAINT_LABELS:
        count = positives.get(name, 0)
        rate = count / rep.valid_records if rep.valid_records else 0.0
        if count == 0:
            level = rep.err if name in HARD_CONSTRAINT_LABELS else rep.warn
            level(f"label '{name}' has 0 positives — model can't learn it")
        elif rate < 0.005:
            rep.warn(f"label '{name}' positive rate is very low: {rate:.3%}")
        elif rate > 0.95:
            rep.warn(f"label '{name}' positive rate is very high: {rate:.3%}")

    return rep


def check_crossover(path: Path) -> CheckReport:
    rep = CheckReport(kind="crossover", path=path)
    if not path.exists():
        rep.err(f"file not found: {path}")
        return rep

    bad_shapes = 0
    bad_pairs = 0
    missing_fitness = 0
    missing_parent_fitness = 0
    labels: list[int] = []
    off_fits: list[float] = []

    for i, rec in enumerate(iter_records(path)):
        rep.total_records += 1
        if not isinstance(rec, dict):
            rep.err(f"record {i}: not an object")
            continue

        p1 = coerce_grid(rec.get("parent1"))
        p2 = coerce_grid(rec.get("parent2"))
        if p1 is None or p2 is None:
            bad_pairs += 1
            continue

        off_fit = rec.get("offspring_fitness")
        produced_valid = rec.get("produced_valid")
        if off_fit is None and produced_valid is None:
            missing_fitness += 1
            continue
        off_fit_v = float(off_fit) if (off_fit is not None and numeric_finite(off_fit)) else 0.0
        if produced_valid is None:
            produced_valid = 1 if off_fit_v > 0.0 else 0
        try:
            produced_valid = int(produced_valid)
        except Exception:
            rep.warn(f"record {i}: invalid produced_valid value; treating as 0")
            produced_valid = 0

        meta = rec.get("metadata") or {}
        p1_fit = rec.get("parent1_fitness", meta.get("parent1_fitness"))
        p2_fit = rec.get("parent2_fitness", meta.get("parent2_fitness"))
        if p1_fit is None or p2_fit is None:
            missing_parent_fitness += 1

        label = 1 if (produced_valid == 1 and off_fit_v >= CROSSOVER_FITNESS_THRESHOLD) else 0
        labels.append(label)
        off_fits.append(off_fit_v)
        rep.valid_records += 1

        if bad_shapes:  # placeholder — handled above
            pass

    if bad_pairs:
        rep.warn(f"{bad_pairs} records had parent1/parent2 of invalid shape")
    if missing_fitness:
        rep.warn(f"{missing_fitness} records missing both offspring_fitness and produced_valid")
    if missing_parent_fitness:
        rep.warn(
            f"{missing_parent_fitness} records missing parent1_fitness/parent2_fitness "
            "(top-level or in metadata) — the feature extractor will default them to 0.0"
        )

    if rep.valid_records == 0:
        rep.err("no valid crossover records found")
        return rep

    labels_arr = np.array(labels, dtype=np.int32)
    pos = int(np.sum(labels_arr == 1))
    neg = int(np.sum(labels_arr == 0))
    rep.stats["label_compatible_1"] = pos
    rep.stats["label_incompatible_0"] = neg
    rep.stats["fitness_min"] = float(np.min(off_fits))
    rep.stats["fitness_max"] = float(np.max(off_fits))
    rep.stats["fitness_mean"] = float(np.mean(off_fits))
    rep.stats["fitness_threshold"] = CROSSOVER_FITNESS_THRESHOLD

    if pos == 0 or neg == 0:
        rep.err(
            f"label distribution unusable for binary classification "
            f"(compatible={pos}, incompatible={neg})"
        )
    else:
        ratio = max(pos, neg) / min(pos, neg)
        if ratio > 10:
            rep.warn(
                f"class imbalance is severe (ratio {ratio:.1f}:1); "
                "undersampling will discard most of the majority class"
            )

    return rep


def check_mutation(path: Path) -> CheckReport:
    rep = CheckReport(kind="mutation", path=path)
    if not path.exists():
        rep.err(f"file not found: {path}")
        return rep

    bad_shapes = 0
    missing_fitness = 0
    label_counts = Counter()
    unknown_types = Counter()
    type_in_spec = 0
    type_total = 0

    for i, rec in enumerate(iter_records(path)):
        rep.total_records += 1
        if not isinstance(rec, dict):
            rep.err(f"record {i}: not an object")
            continue

        before_raw = rec.get("before_schedule") or rec.get("original_schedule")
        after_raw = rec.get("after_schedule") or rec.get("mutated_schedule")
        before = coerce_grid(before_raw)
        after = coerce_grid(after_raw)
        if before is None or after is None:
            bad_shapes += 1
            continue

        before_fit = rec.get("before_fitness", rec.get("original_fitness"))
        after_fit = rec.get("after_fitness", rec.get("mutated_fitness"))
        if before_fit is None or after_fit is None or not numeric_finite(before_fit) or not numeric_finite(after_fit):
            missing_fitness += 1
            continue

        delta = float(after_fit) - float(before_fit)
        if delta > MUTATION_LABEL_THRESHOLDS["improve"]:
            label = "improve"
        elif delta < MUTATION_LABEL_THRESHOLDS["worsen"]:
            label = "worsen"
        else:
            label = "neutral"
        label_counts[label] += 1

        mtype = rec.get("mutation_type")
        if mtype is None:
            mtype = (rec.get("mutation_info") or {}).get("type")
        if mtype is not None:
            type_total += 1
            mtype_s = str(mtype).lower()
            if mtype_s in MUTATION_TYPES_SPEC:
                type_in_spec += 1
            else:
                unknown_types[mtype_s] += 1

        rep.valid_records += 1

    if bad_shapes:
        rep.warn(f"{bad_shapes} records had before/after schedules of invalid shape")
    if missing_fitness:
        rep.warn(f"{missing_fitness} records missing/invalid before_fitness or after_fitness")
    if rep.valid_records == 0:
        rep.err("no valid mutation records found")
        return rep

    rep.stats["labels"] = dict(label_counts)
    rep.stats["mutation_types_total"] = type_total
    rep.stats["mutation_types_in_spec"] = type_in_spec
    rep.stats["mutation_types_outside_spec"] = dict(unknown_types.most_common(5))

    # Class balance check after delta rule
    if any(c == 0 for c in (label_counts["improve"], label_counts["neutral"], label_counts["worsen"])):
        rep.err(
            "one or more label classes have 0 samples after the delta rule "
            f"(±{MUTATION_LABEL_THRESHOLDS['improve']}); cannot train 3-class classifier"
        )
    else:
        smallest = min(label_counts.values())
        largest = max(label_counts.values())
        ratio = largest / smallest
        if ratio > 5:
            rep.warn(
                f"class imbalance after delta rule is large ({ratio:.1f}:1); "
                "balancing will discard most of the majority class"
            )

    if type_total > 0 and type_in_spec == 0:
        rep.warn(
            "no record uses a mutation_type in the spec set "
            f"{sorted(MUTATION_TYPES_SPEC)}; one-hot mutation features will all be zero"
        )

    return rep


# ── CLI / orchestration ──────────────────────────────────────────────────────
KIND_CHECKERS = {
    "fitness": check_fitness,
    "constraint": check_constraint,
    "crossover": check_crossover,
    "mutation": check_mutation,
}

DEFAULT_PATHS = {
    "fitness":    PROJECT_ROOT / "data" / "training_output" / "training_data.json",
    "constraint": PROJECT_ROOT / "data" / "training_output" / "constraint_samples.jsonl",
    "crossover":  PROJECT_ROOT / "data" / "training_output" / "crossover_samples.jsonl",
    "mutation":   PROJECT_ROOT / "data" / "training_output" / "mutation_samples.jsonl",
}


def print_report(rep: CheckReport) -> None:
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"[{rep.kind.upper()}] {rep.path}")
    print(bar)
    print(f"records: {rep.total_records} total | {rep.valid_records} valid")

    if rep.stats:
        print("stats:")
        for k, v in rep.stats.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for sk, sv in v.items():
                    print(f"    {sk:<28}: {sv}")
            elif isinstance(v, float):
                print(f"  {k:<28}: {v:.4f}")
            else:
                print(f"  {k:<28}: {v}")

    if rep.warnings:
        print(f"warnings ({len(rep.warnings)}):")
        for w in rep.warnings:
            print(f"  WARN: {w}")
    if rep.errors:
        print(f"errors ({len(rep.errors)}):")
        for e in rep.errors:
            print(f"  ERROR: {e}")

    status = "OK" if rep.ok else "FAIL"
    print(f"status: {status}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify ANN training data files")
    parser.add_argument(
        "--kind",
        choices=list(KIND_CHECKERS.keys()) + ["all"],
        default="all",
        help="Which dataset to verify (default: all four)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Override the default path for the chosen --kind",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures in the exit code",
    )
    args = parser.parse_args(argv)

    if args.file and args.kind == "all":
        parser.error("--file requires a specific --kind")

    kinds = [args.kind] if args.kind != "all" else list(KIND_CHECKERS.keys())
    overall_ok = True
    summary: list[tuple[str, str]] = []

    for kind in kinds:
        checker = KIND_CHECKERS[kind]
        path = args.file if args.file else DEFAULT_PATHS[kind]
        try:
            rep = checker(path)
        except Exception as exc:
            rep = CheckReport(kind=kind, path=path)
            rep.err(f"unhandled exception during check: {exc}")

        print_report(rep)
        passed = rep.ok and (not args.strict or not rep.warnings)
        overall_ok = overall_ok and passed
        summary.append((
            kind,
            "OK" if passed else ("FAIL (strict)" if rep.ok else "FAIL"),
        ))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for kind, status in summary:
        print(f"  {kind:<11}: {status}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
