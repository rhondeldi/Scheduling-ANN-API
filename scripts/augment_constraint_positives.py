"""
Synthesize positive samples for the hard-conflict constraint labels.

The GA's mutation operators preserve feasibility, so the natural data has zero
positives for ``instructor_conflict`` and ``room_conflict`` — the constraint
classifier has no signal to learn those labels.

This script fabricates plausible positives by:

  1. Streaming the existing constraint samples (JSONL) twice.
  2. First pass: reservoir-sample a small pool of clean rows (rows with no
     hard-conflict positives) to use as templates.
  3. Second pass: copy every original row to the output, attaching a
     zero ``cross_section`` dict if missing. Also writes the synthetic
     rows generated from the template pool.

Synthetic rows:
  - keep the template's ``section_schedule`` unchanged,
  - mark either ``instructor_conflict`` or ``room_conflict`` true (or both),
  - attach a ``cross_section`` dict whose conflict counts are consistent
    with the chosen label.

Memory footprint is O(reservoir-size), not O(input-rows) — safe for the
multi-GB JSONL files the GA emits.

Usage:
    python scripts/augment_constraint_positives.py
    python scripts/augment_constraint_positives.py \
        --input data/training_output/constraint_samples.jsonl \
        --output data/training_output/constraint_samples.augmented.jsonl \
        --per-label 6000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config

N_DAYS = config.N_WEEKLY_SCHOOL_DAYS
N_SLOTS = config.N_DAILY_TIME_SLOTS

DEFAULT_INPUT = PROJECT_ROOT / "data" / "training_output" / "constraint_samples.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "training_output" / "constraint_samples.augmented.jsonl"

HARD_LABELS = ("instructor_conflict", "room_conflict")
ALL_LABELS = (
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
)

CROSS_KEYS = (
    "total_instructor_conflicts",
    "total_room_conflicts",
    "max_instructor_conflicts_in_one_slot",
    "max_room_conflicts_in_one_slot",
    "slots_with_any_conflict",
)


def _occupied_slot_count(section: list) -> int:
    """Number of slots in the section where subject_id > 0."""
    try:
        arr = np.asarray(section, dtype=np.int32)
    except Exception:
        return 0
    if arr.ndim != 3:
        return 0
    return int(np.sum(arr[:, :, 0] > 0))


def _zero_cross_section() -> dict:
    return {k: 0 for k in CROSS_KEYS}


def _is_clean_hard(rec: dict) -> bool:
    """A record qualifies as a 'clean' source if both hard labels are false."""
    v = rec.get("violations") or {}
    if not isinstance(v, dict):
        return False
    return not (bool(v.get("instructor_conflict")) or bool(v.get("room_conflict")))


def _synthesize_aggregates(
    occupied: int,
    label: str,
    rng: random.Random,
) -> dict:
    """Build a cross_section dict consistent with the chosen positive label."""
    if occupied <= 0:
        agg = _zero_cross_section()
        if label in ("instructor_conflict", "both"):
            agg["total_instructor_conflicts"] = 1
            agg["max_instructor_conflicts_in_one_slot"] = 1
        if label in ("room_conflict", "both"):
            agg["total_room_conflicts"] = 1
            agg["max_room_conflicts_in_one_slot"] = 1
        agg["slots_with_any_conflict"] = 1
        return agg

    agg = _zero_cross_section()
    distinct_slots: set[int] = set()
    n_target_slots = rng.randint(1, min(3, occupied))

    for slot_id in range(n_target_slots):
        if label in ("instructor_conflict", "both"):
            others = rng.randint(1, 3)
            agg["total_instructor_conflicts"] += others
            agg["max_instructor_conflicts_in_one_slot"] = max(
                agg["max_instructor_conflicts_in_one_slot"], others
            )
            distinct_slots.add(slot_id)
        if label in ("room_conflict", "both"):
            others = rng.randint(1, 3)
            agg["total_room_conflicts"] += others
            agg["max_room_conflicts_in_one_slot"] = max(
                agg["max_room_conflicts_in_one_slot"], others
            )
            distinct_slots.add(slot_id)

    agg["slots_with_any_conflict"] = len(distinct_slots)
    return agg


def _make_violations(template: dict | None, positive_labels: Iterable[str]) -> dict:
    """Copy soft labels from template, set chosen positives true, ensure all keys exist."""
    out = {label: False for label in ALL_LABELS}
    if isinstance(template, dict):
        for label in ALL_LABELS:
            if label in HARD_LABELS:
                continue
            if bool(template.get(label)):
                out[label] = True
    for label in positive_labels:
        out[label] = True
    return out


def _ensure_cross_section(rec: dict) -> dict:
    """Add a zero cross_section dict if the record doesn't already have one."""
    if "cross_section" in rec and isinstance(rec["cross_section"], dict):
        return rec
    return {**rec, "cross_section": _zero_cross_section()}


def _stream_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _reservoir_sample_templates(
    path: Path,
    reservoir_size: int,
    rng: random.Random,
) -> tuple[list[dict], int, int]:
    """Single streaming pass that returns up to ``reservoir_size`` clean rows.

    Also returns (total_input_rows, total_clean_rows) for the final stats.
    """
    reservoir: list[dict] = []
    total = 0
    clean = 0
    for rec in _stream_records(path):
        if not isinstance(rec, dict):
            continue
        total += 1
        if not _is_clean_hard(rec):
            continue
        if not isinstance(rec.get("section_schedule"), list):
            continue
        clean += 1
        if len(reservoir) < reservoir_size:
            reservoir.append(rec)
        else:
            # Vitter's algorithm R — uniform sample
            j = rng.randint(0, clean - 1)
            if j < reservoir_size:
                reservoir[j] = rec
    return reservoir, total, clean


def augment(
    input_path: Path,
    output_path: Path,
    per_label: int,
    both_ratio: float,
    reservoir_size: int,
    seed: int,
) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    rng = random.Random(seed)

    # ── pass 1 — reservoir-sample template rows ──────────────────────────────
    templates, total_in, total_clean = _reservoir_sample_templates(
        input_path, reservoir_size, rng,
    )
    if not templates:
        raise ValueError(
            f"no clean source rows (both hard labels false) found in {input_path}"
        )

    plan: list[str] = (
        ["instructor_conflict"] * per_label
        + ["room_conflict"] * per_label
        + ["both"] * int(per_label * both_ratio)
    )
    rng.shuffle(plan)

    # ── write — synthetics first, then stream-copy originals ─────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written_synthetic = 0
    written_originals = 0
    enriched_originals = 0

    with output_path.open("w", encoding="utf-8") as out:
        for label in plan:
            template = rng.choice(templates)
            section = template["section_schedule"]
            occupied = _occupied_slot_count(section)
            positives = (
                ("instructor_conflict", "room_conflict")
                if label == "both"
                else (label,)
            )
            synth = {
                "section_schedule": section,
                "department_id": template.get("department_id", 0),
                "violations": _make_violations(template.get("violations"), positives),
                "cross_section": _synthesize_aggregates(occupied, label, rng),
                "_synthetic": True,
            }
            out.write(json.dumps(synth, ensure_ascii=False))
            out.write("\n")
            written_synthetic += 1

        # ── pass 2 — stream-copy originals with cross_section attached ───────
        for rec in _stream_records(input_path):
            if not isinstance(rec, dict):
                continue
            if "cross_section" not in rec:
                enriched_originals += 1
            rec = _ensure_cross_section(rec)
            out.write(json.dumps(rec, ensure_ascii=False))
            out.write("\n")
            written_originals += 1

    return {
        "input_rows": total_in,
        "clean_source_pool": total_clean,
        "template_reservoir": len(templates),
        "synthetic_rows": written_synthetic,
        "originals_copied": written_originals,
        "originals_enriched_with_zero_cross_section": enriched_originals,
        "output_rows": written_synthetic + written_originals,
        "output_path": str(output_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synthesize hard-conflict positives for the constraint classifier",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help=f"input JSONL (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"output JSONL (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--per-label", type=int, default=6000,
                        help="synthetic positives per hard label (default: 6000)")
    parser.add_argument("--both-ratio", type=float, default=0.15,
                        help="fraction of per-label that have BOTH hard labels positive (default: 0.15)")
    parser.add_argument("--reservoir-size", type=int, default=20000,
                        help="max clean templates kept in memory (default: 20000)")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED,
                        help="RNG seed")
    args = parser.parse_args(argv)

    if args.per_label <= 0:
        parser.error("--per-label must be > 0")
    if not (0.0 <= args.both_ratio <= 1.0):
        parser.error("--both-ratio must be in [0, 1]")
    if args.reservoir_size <= 0:
        parser.error("--reservoir-size must be > 0")

    stats = augment(
        input_path=args.input,
        output_path=args.output,
        per_label=args.per_label,
        both_ratio=args.both_ratio,
        reservoir_size=args.reservoir_size,
        seed=args.seed,
    )

    print("=" * 70)
    print("CONSTRAINT AUGMENTATION — DONE")
    print("=" * 70)
    for k, v in stats.items():
        print(f"  {k:<42}: {v}")
    print("\nNext: re-run the verifier against the augmented file:")
    print(f"  python scripts/verify_training_data.py --kind constraint "
          f"--file {stats['output_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
