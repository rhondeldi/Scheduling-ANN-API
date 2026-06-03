"""
Generate synthetic schedule variants from existing manual schedules.

Fitness function mirrors the Go backend exactly:
  GeneticAlgorithm/fitness.go → MeasureWeekTimeTableBasicFitness

Usage:
    python scripts/generate_synthetic_variants.py \
        --input  data/manual_schedules_with_fitness.json \
        --output data/training_data.json \
        --target 5000 \
        --seed   42
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── allow running from repo root or scripts/ ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
#  FITNESS FUNCTION  (exact Go backend parity — no external import needed)
# ══════════════════════════════════════════════════════════════════════════════

N_DAYS               = 6
N_SLOTS              = 24
N_ATTRS              = 3
N_HOUR_SLOTS         = 2        # 2 slots = 1 hour  (each slot = 30 min)
PREFERRED_MAX_HOURS  = 10.0     # PREFERRED_MAX_CLASS_HOUR_PER_DAY


from scripts.calculate_fitness_for_historical import (
    measure_week_timetable_basic_fitness as measure_week_fitness,
)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_schedules(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    items = raw.get("schedules", raw.get("data", raw)) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError(f"Unsupported format in {path}")
    return [i for i in items if isinstance(i, dict)]


def normalize_week_schedule(raw_ws: Any) -> List[List[List[int]]]:
    ws = [[[0, 0, 0] for _ in range(N_SLOTS)] for _ in range(N_DAYS)]
    if not isinstance(raw_ws, list):
        return ws
    for d in range(min(N_DAYS, len(raw_ws))):
        day = raw_ws[d]
        if not isinstance(day, list):
            continue
        for s in range(min(N_SLOTS, len(day))):
            slot = day[s]
            if isinstance(slot, (list, tuple)) and len(slot) >= N_ATTRS:
                try:
                    ws[d][s] = [int(slot[0]), int(slot[1]), int(slot[2])]
                except Exception:
                    ws[d][s] = [0, 0, 0]
    return ws


def extract_week_schedule(sample: Dict[str, Any]) -> List[List[List[int]]]:
    if isinstance(sample.get("schedule"), dict):
        raw = sample["schedule"].get("week_schedule")
    else:
        raw = sample.get("week_schedule")
    return normalize_week_schedule(raw)


def flatten(ws: List[List[List[int]]]) -> List[List[int]]:
    return [slot for day in ws for slot in day]


def unflatten(flat: List[List[int]]) -> List[List[List[int]]]:
    out, idx = [], 0
    for _ in range(N_DAYS):
        day = [flat[idx + s] for s in range(N_SLOTS)]
        idx += N_SLOTS
        out.append(day)
    return out


def build_sample(
    schedule_id: str,
    week_schedule: List[List[List[int]]],
    fitness: float,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "schedule":    {"week_schedule": week_schedule},
        "fitness":     float(fitness),
        "schedule_id": schedule_id,
        "metadata":    metadata,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MUTATION OPERATORS
#  Each operator preserves the subject/instructor/room triple — it only
#  moves or swaps class blocks, never invents new IDs.
# ══════════════════════════════════════════════════════════════════════════════

def _occupied(flat: List[List[int]]) -> List[int]:
    return [i for i, s in enumerate(flat) if s[0] > 0]

def _empty(flat: List[List[int]]) -> List[int]:
    return [i for i, s in enumerate(flat) if s[0] == 0]


def mutate_swap_slots(flat: List[List[int]], rng: random.Random) -> Tuple[List[List[int]], dict]:
    """Swap two occupied slots (changes time placement of two classes)."""
    occ = _occupied(flat)
    if len(occ) < 2:
        return flat, {"type": "noop"}
    i, j = rng.sample(occ, 2)
    flat[i], flat[j] = flat[j], flat[i]
    return flat, {"type": "swap_slots", "i": i, "j": j}


def mutate_move_class(flat: List[List[int]], rng: random.Random) -> Tuple[List[List[int]], dict]:
    """Move one occupied slot to a random empty slot."""
    occ = _occupied(flat)
    emp = _empty(flat)
    if not occ or not emp:
        return flat, {"type": "noop"}
    src = rng.choice(occ)
    dst = rng.choice(emp)
    flat[dst] = flat[src]
    flat[src] = [0, 0, 0]
    return flat, {"type": "move_class", "src": src, "dst": dst}


def mutate_swap_days(ws: List[List[List[int]]], rng: random.Random) -> Tuple[List[List[List[int]]], dict]:
    """Swap all slots of two randomly chosen days."""
    d1, d2 = rng.sample(range(N_DAYS), 2)
    ws[d1], ws[d2] = ws[d2], ws[d1]
    return ws, {"type": "swap_days", "d1": d1, "d2": d2}


def mutate_shift_day_slots(ws: List[List[List[int]]], rng: random.Random) -> Tuple[List[List[List[int]]], dict]:
    """
    Shift all classes in one day forward or backward by 1–3 slots,
    wrapping classes that fall off the edge to the other side.
    This nudges start times and lunch/5pm conditions.
    """
    day = rng.randrange(N_DAYS)
    shift = rng.choice([-3, -2, -1, 1, 2, 3])
    ws[day] = ws[day][-shift:] + ws[day][:-shift] if shift < 0 else ws[day][shift:] + ws[day][:shift]
    return ws, {"type": "shift_day", "day": day, "shift": shift}


def mutate_clear_day(ws: List[List[List[int]]], rng: random.Random) -> Tuple[List[List[List[int]]], dict]:
    """
    Clear a random day that has classes (simulates a section having a free day).
    Only clears if at least 2 other days have classes, so schedule stays non-empty.
    """
    days_with_class = [d for d in range(N_DAYS) if any(ws[d][s][0] > 0 for s in range(N_SLOTS))]
    if len(days_with_class) <= 2:
        return ws, {"type": "noop"}
    day = rng.choice(days_with_class)
    ws[day] = [[0, 0, 0] for _ in range(N_SLOTS)]
    return ws, {"type": "clear_day", "day": day}


def apply_mutations(
    ws: List[List[List[int]]],
    rng: random.Random,
    n_mutations: int,
) -> Tuple[List[List[List[int]]], List[dict]]:
    """
    Apply n_mutations chosen from the operator pool.
    Operators are weighted to produce a realistic spread of fitness outcomes.
    """
    ws = copy.deepcopy(ws)
    log = []

    # weights: shift and swap_days change fitness conditions most dramatically
    operators = ["swap_slots", "move_class", "swap_days", "shift_day", "clear_day"]
    weights   = [0.20,         0.20,         0.25,        0.25,        0.10]

    for _ in range(n_mutations):
        op = rng.choices(operators, weights=weights, k=1)[0]

        if op == "swap_slots":
            flat = flatten(ws)
            flat, info = mutate_swap_slots(flat, rng)
            ws = unflatten(flat)
        elif op == "move_class":
            flat = flatten(ws)
            flat, info = mutate_move_class(flat, rng)
            ws = unflatten(flat)
        elif op == "swap_days":
            ws, info = mutate_swap_days(ws, rng)
        elif op == "shift_day":
            ws, info = mutate_shift_day_slots(ws, rng)
        elif op == "clear_day":
            ws, info = mutate_clear_day(ws, rng)
        else:
            info = {"type": "noop"}

        log.append(info)

    return ws, log


# ══════════════════════════════════════════════════════════════════════════════
#  VARIANT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def fitness_bucket(f: float) -> str:
    if f <= -20:   return "empty"
    if f <    0:   return "poor"
    if f <    7:   return "fair"
    if f <   13:   return "good"
    return "excellent"


def generate_variants(
    base_samples: List[Dict[str, Any]],
    target_count: int,
    seed: int,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)

    # ── normalize all base schedules ──────────────────────────────────────────
    normalized: List[Tuple[str, List[List[List[int]]], float]] = []
    skipped = 0
    for idx, sample in enumerate(base_samples):
        ws  = extract_week_schedule(sample)
        fit = sample.get("fitness")
        if not isinstance(fit, (int, float)):
            fit = measure_week_fitness(ws)
        fit = float(fit)

        # skip genuinely empty schedules (-24.0) — they pollute training data
        if fit <= -24.0:
            skipped += 1
            continue

        sid = sample.get("id") or sample.get("schedule_id") or f"schedule_{idx}"
        normalized.append((sid, ws, fit))

    if not normalized:
        raise ValueError("No valid (non-empty) schedules found in input data.")

    if verbose:
        print(f"  Base schedules loaded  : {len(normalized)}  (skipped {skipped} empty)")

    # ── seed output with originals ────────────────────────────────────────────
    output: List[Dict[str, Any]] = []
    for sid, ws, fit in normalized:
        output.append(build_sample(sid, ws, fit, {"source": "manual", "base_id": sid}))

    bucket_counts: Counter = Counter(fitness_bucket(f) for _, _, f in normalized)
    variant_idx = 0

    # ── Phase 1: oversample the 0–7 "fair" bucket until it matches "excellent" ─
    # This directly addresses SMAPE weakness on near-zero scores by giving the
    # model enough signal in that range before filling out the rest of the dataset.
    #
    # Strategy: take any base schedule with fitness >= 7 (good/excellent) and
    # apply 2–4 mutations. shift_day and swap_days are the operators most likely
    # to push a score across the lunch / after-5pm boundaries that separate
    # "good" from "fair", so we raise their weights here.
    fair_target = max(bucket_counts["excellent"], bucket_counts["good"], 200)
    fair_oversample_attempts = 0
    MAX_FAIR_ATTEMPTS = fair_target * 30  # safety exit

    good_or_excellent = [(sid, ws, f) for sid, ws, f in normalized if f >= 7]
    if not good_or_excellent:
        good_or_excellent = normalized  # fallback if dataset is all poor

    if verbose:
        print(f"\n  🎯 Phase 1: oversampling 'fair' bucket to ~{fair_target} samples ...")

    while bucket_counts["fair"] < fair_target and fair_oversample_attempts < MAX_FAIR_ATTEMPTS:
        fair_oversample_attempts += 1
        base_id, base_ws, base_fit = rng.choice(good_or_excellent)

        # 2–4 mutations weighted toward the operators that cross fitness boundaries
        n_mut   = rng.randint(2, 4)
        variant = copy.deepcopy(base_ws)
        mut_log = []

        boundary_ops  = ["shift_day", "swap_days"]
        boundary_w    = [0.50, 0.50]
        other_ops     = ["swap_slots", "move_class", "clear_day"]
        other_w       = [0.30, 0.30, 0.40]

        for step in range(n_mut):
            # first mutation always a boundary op; rest random
            if step == 0:
                op = rng.choices(boundary_ops, weights=boundary_w, k=1)[0]
            else:
                op = rng.choices(other_ops, weights=other_w, k=1)[0]

            if op == "shift_day":
                variant, info = mutate_shift_day_slots(variant, rng)
            elif op == "swap_days":
                variant, info = mutate_swap_days(variant, rng)
            elif op == "swap_slots":
                flat = flatten(variant)
                flat, info = mutate_swap_slots(flat, rng)
                variant = unflatten(flat)
            elif op == "move_class":
                flat = flatten(variant)
                flat, info = mutate_move_class(flat, rng)
                variant = unflatten(flat)
            elif op == "clear_day":
                variant, info = mutate_clear_day(variant, rng)
            else:
                info = {"type": "noop"}
            mut_log.append(info)

        fit = measure_week_fitness(variant)

        # only keep if it landed in the fair range (0–7)
        if not (0.0 <= fit < 7.0):
            continue

        new_id = f"{base_id}_fair_{variant_idx}"
        output.append(build_sample(
            new_id, variant, fit,
            {
                "source":      "synthetic_fair_oversample",
                "base_id":     base_id,
                "base_fit":    base_fit,
                "n_mutations": len(mut_log),
                "mutations":   mut_log,
            },
        ))
        bucket_counts["fair"] += 1
        variant_idx += 1

    if verbose:
        print(f"     fair bucket now : {bucket_counts['fair']}  (attempts: {fair_oversample_attempts})")

    # ── Phase 2: fill remaining quota with standard adaptive variants ──────────
    if verbose:
        print(f"\n  🧬 Phase 2: filling remaining {max(0, target_count - len(output))} samples ...")

    while len(output) < target_count:
        base_id, base_ws, base_fit = rng.choice(normalized)

        if base_fit >= 13:
            n_mut = rng.randint(1, 2)
        elif base_fit >= 7:
            n_mut = rng.randint(1, 4)
        else:
            n_mut = rng.randint(2, 6)

        variant, mut_log = apply_mutations(base_ws, rng, n_mut)
        fit = measure_week_fitness(variant)

        if fit <= -24.0:
            continue

        new_id = f"{base_id}_syn_{variant_idx}"
        output.append(build_sample(
            new_id, variant, fit,
            {
                "source":      "synthetic_variant",
                "base_id":     base_id,
                "base_fit":    base_fit,
                "n_mutations": len(mut_log),
                "mutations":   mut_log,
            },
        ))
        bucket_counts[fitness_bucket(fit)] += 1
        variant_idx += 1

    result = output[:target_count]

    if verbose:
        fitnesses = [s["fitness"] for s in result]
        print(f"\n  📊 Final dataset ({len(result)} samples)")
        print(f"     Fitness range  : {min(fitnesses):.4f} – {max(fitnesses):.4f}")
        print(f"     Mean / Median  : {statistics.mean(fitnesses):.4f} / {statistics.median(fitnesses):.4f}")
        print(f"     Std deviation  : {statistics.pstdev(fitnesses):.4f}")
        print(f"\n  🪣  Bucket distribution:")
        for bucket in ["excellent", "good", "fair", "poor", "empty"]:
            c = bucket_counts[bucket]
            pct = c / len(result) * 100
            print(f"     {bucket:<10} : {c:>5}  ({pct:.1f}%)")

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic schedule variants for ANN training")
    parser.add_argument("--input",  default="data/manual_schedules_with_fitness.json",
                        help="Path to scored manual schedules JSON")
    parser.add_argument("--output", default="data/training_data.json",
                        help="Output path for training data JSON")
    parser.add_argument("--target", type=int, default=5000,
                        help="Total number of training samples to generate (default: 5000)")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args(argv)

    input_path  = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        print(f"❌  Input file not found: {input_path}", file=sys.stderr)
        return 1

    print(f"📂  Loading base schedules from: {input_path}")
    base_samples = load_schedules(input_path)
    print(f"✓   Loaded {len(base_samples)} entries")

    print(f"\n🧬  Generating up to {args.target} training samples (seed={args.seed}) ...")
    result = generate_variants(base_samples, args.target, args.seed, verbose=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n💾  Saved {len(result)} samples → {output_path}")
    print("\n✅  Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
