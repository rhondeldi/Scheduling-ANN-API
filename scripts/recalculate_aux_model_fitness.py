"""
Recalculate fitness fields used by the auxiliary ANN model datasets.

This script uses the same calculator as calculate_fitness_for_historical.py,
which is the Python port of the Go backend fitness function.

Usage:
    python scripts/recalculate_aux_model_fitness.py
    python scripts/recalculate_aux_model_fitness.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.calculate_fitness_for_historical import (  # noqa: E402
    N_DAILY_TIME_SLOTS,
    N_WEEKLY_SCHOOL_DAYS,
    measure_week_timetable_basic_fitness,
)


DATASETS = {
    "mutation": PROJECT_ROOT / "data" / "mutation_data.json",
    "crossover": PROJECT_ROOT / "data" / "crossover_data.json",
    "constraint": PROJECT_ROOT / "data" / "constraint_data.json",
}


def read_samples(path: Path) -> tuple[Any, List[Dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    samples = raw.get("schedules", raw.get("data", raw)) if isinstance(raw, dict) else raw
    if not isinstance(samples, list):
        raise ValueError(f"Unsupported dataset format: {path}")
    return raw, samples


def write_samples(path: Path, raw: Any) -> None:
    path.write_text(json.dumps(raw, indent=2), encoding="utf-8")


def week_schedule(schedule: Dict[str, Any] | None) -> List[List[List[int]]] | None:
    if not isinstance(schedule, dict):
        return None
    value = schedule.get("week_schedule")
    return value if isinstance(value, list) else None


def flatten(week: List[List[List[int]]]) -> List[List[int]]:
    return [slot for day in week for slot in day]


def unflatten(slots: List[List[int]]) -> List[List[List[int]]]:
    return [
        slots[day * N_DAILY_TIME_SLOTS : (day + 1) * N_DAILY_TIME_SLOTS]
        for day in range(N_WEEKLY_SCHOOL_DAYS)
    ]


def single_point_offspring(
    parent1: List[List[List[int]]],
    parent2: List[List[List[int]]],
    crossover_point: int,
) -> List[List[List[int]]]:
    slots_per_week = N_WEEKLY_SCHOOL_DAYS * N_DAILY_TIME_SLOTS
    split = max(0, min(slots_per_week, int(crossover_point)))
    p1_slots = flatten(parent1)
    p2_slots = flatten(parent2)
    return unflatten(p1_slots[:split] + p2_slots[split:])


def fitness(schedule: Dict[str, Any] | None) -> float | None:
    ws = week_schedule(schedule)
    if ws is None:
        return None
    return float(measure_week_timetable_basic_fitness(ws))


def update_if_changed(target: Dict[str, Any], key: str, value: float | None) -> bool:
    if value is None:
        return False
    old = target.get(key)
    if isinstance(old, (int, float)) and abs(float(old) - value) < 1e-9:
        return False
    target[key] = value
    return True


def recalculate_mutation(path: Path) -> tuple[int, int]:
    raw, samples = read_samples(path)
    changed = 0

    for sample in samples:
        original = fitness(sample.get("original_schedule") or sample.get("current_schedule"))
        mutated = fitness(sample.get("mutated_schedule"))

        did_change = False
        did_change |= update_if_changed(sample, "original_fitness", original)
        did_change |= update_if_changed(sample, "mutated_fitness", mutated)

        if original is not None and mutated is not None:
            delta = mutated - original
            impact = "improve" if delta > 0 else "worsen" if delta < 0 else "neutral"
            did_change |= update_if_changed(sample.setdefault("mutation_info", {}), "fitness_delta", delta)
            if sample.get("impact") != impact:
                sample["impact"] = impact
                did_change = True

        changed += int(did_change)

    write_samples(path, raw)
    return changed, len(samples)


def recalculate_crossover(path: Path) -> tuple[int, int]:
    raw, samples = read_samples(path)
    changed = 0

    for sample in samples:
        p1_fit = fitness(sample.get("parent1"))
        p2_fit = fitness(sample.get("parent2"))

        did_change = False
        metadata = sample.setdefault("metadata", {})
        did_change |= update_if_changed(metadata, "parent1_fitness", p1_fit)
        did_change |= update_if_changed(metadata, "parent2_fitness", p2_fit)

        if p1_fit is not None and p2_fit is not None:
            baseline = (p1_fit + p2_fit) / 2.0
            did_change |= update_if_changed(metadata, "baseline_fitness", baseline)
        else:
            baseline = None

        offspring_schedule = sample.get("offspring")
        if not isinstance(offspring_schedule, dict):
            parent1_ws = week_schedule(sample.get("parent1"))
            parent2_ws = week_schedule(sample.get("parent2"))
            if parent1_ws is not None and parent2_ws is not None:
                offspring_schedule = {
                    "week_schedule": single_point_offspring(
                        parent1_ws,
                        parent2_ws,
                        int(sample.get("crossover_point", 0)),
                    )
                }

        offspring_fit = fitness(offspring_schedule)
        did_change |= update_if_changed(sample, "offspring_fitness", offspring_fit)

        if offspring_fit is not None and baseline is not None:
            did_change |= update_if_changed(metadata, "improvement", offspring_fit - baseline)

        changed += int(did_change)

    write_samples(path, raw)
    return changed, len(samples)


def recalculate_constraint(path: Path) -> tuple[int, int]:
    raw, samples = read_samples(path)
    changed = 0

    for sample in samples:
        value = fitness(sample.get("schedule"))
        did_change = update_if_changed(sample.setdefault("metadata", {}), "fitness", value)
        changed += int(did_change)

    write_samples(path, raw)
    return changed, len(samples)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recalculate auxiliary model fitness fields")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without keeping file updates")
    args = parser.parse_args(argv)

    original_text = {name: path.read_text(encoding="utf-8-sig") for name, path in DATASETS.items()}

    results = {
        "mutation": recalculate_mutation(DATASETS["mutation"]),
        "crossover": recalculate_crossover(DATASETS["crossover"]),
        "constraint": recalculate_constraint(DATASETS["constraint"]),
    }

    if args.dry_run:
        for name, path in DATASETS.items():
            path.write_text(original_text[name], encoding="utf-8")

    for name, (changed, total) in results.items():
        suffix = "would change" if args.dry_run else "changed"
        print(f"{name}: {changed}/{total} samples {suffix}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
