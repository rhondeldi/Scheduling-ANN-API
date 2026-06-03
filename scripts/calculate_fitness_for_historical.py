"""
Calculate Fitness Scores for Historical Schedules

This utility calculates fitness scores for historical/manual schedules that don't have them.
Implements the EXACT same fitness function as the Go backend.

Go source: GeneticAlgorithm/fitness.go
  - MeasureWeekTimeTableBasicFitness
  - MeasureUniSchedBasicFitness

USAGE:
    python calculate_fitness_for_historical.py path/to/schedules.json output.json
"""

import json
import math
import statistics
from pathlib import Path
from typing import List, Dict, Any, Tuple
import sys


# ── Constants (matching Go backend: Resources/Const) ──────────────────────────

N_WEEKLY_SCHOOL_DAYS = 6
N_DAILY_TIME_SLOTS   = 24
N_HOUR_TIME_SLOTS    = 2          # 2 slots per hour (each slot = 30 min)

# ── Fitness constants (matching Go backend: GeneticAlgorithm/fitness.go) ──────

PREFERRED_MAX_CLASS_HOUR_PER_DAY: float = 10.0


# ── Core fitness functions ─────────────────────────────────────────────────────

def reciprocal_distance(actual_hours: float, target_hours: float) -> float:
    """
    Matches Go: reciprocal_distance()
    Output range: (0, 1]
    When actual == target → 1.0
    As |actual - target| → +Inf → 0.0
    """
    return 1.0 / (1.0 + abs(actual_hours - target_hours))


def measure_week_timetable_basic_fitness(week_sched: List[List[List[int]]]) -> float:
    """
    Matches Go: MeasureWeekTimeTableBasicFitness()

    Args:
        week_sched: [N_WEEKLY_SCHOOL_DAYS][N_DAILY_TIME_SLOTS][3]
                    Each slot: [subject_id, instructor_id, room_id]
                    subject_id > 0 means a class is scheduled there.

    Returns:
        float fitness score
    """
    week_sched_fitness = 0.0
    days_with_class    = 0.0

    for day in range(N_WEEKLY_SCHOOL_DAYS):
        if day >= len(week_sched):
            continue

        has_class_after_5pm = False
        has_time_for_lunch  = False
        day_total_hours     = 0.0

        for time_slot in range(N_DAILY_TIME_SLOTS):
            slot = week_sched[day][time_slot] if time_slot < len(week_sched[day]) else [0, 0, 0]
            subject_id = slot[0] if len(slot) > 0 else 0

            # accumulate hours (each slot = 1/N_HOUR_TIME_SLOTS of an hour)
            if subject_id > 0:
                day_total_hours += 1.0 / float(N_HOUR_TIME_SLOTS)

            # any class at slot >= 20 means class after 5 PM
            if time_slot >= 20 and subject_id > 0:
                has_class_after_5pm = True

            # a FREE slot between 8–11 (inclusive) counts as lunch availability
            # Go condition: (time_slot >= 8) && (time_slot <= 11) && subject_id == 0
            if 8 <= time_slot <= 11 and subject_id == 0:
                has_time_for_lunch = True

        # skip days with no classes at all
        if day_total_hours == 0.0:
            continue

        days_with_class += 1.0

        # ── Lunch break reward / punishment ───────────────────────────────────
        # Go: has_time_for_lunch → +8.0, else → -12.0
        if has_time_for_lunch:
            week_sched_fitness += 8.0
        else:
            week_sched_fitness -= 12.0

        # ── After-5pm reward / punishment ────────────────────────────────────
        # Go: has_class_after_5pm → -4.0, else → +4.0
        if has_class_after_5pm:
            week_sched_fitness -= 4.0
        else:
            week_sched_fitness += 4.0


        # ── Preferred max daily hours reward / punishment ────────────────────
        # Go: day_total_hours > PREFERRED_MAX → -3.5, else → +3.5
        if day_total_hours > PREFERRED_MAX_CLASS_HOUR_PER_DAY:
            week_sched_fitness -= 3.5
        else:
            week_sched_fitness += 3.5

        # ── Saturday (last school day) half-day check ────────────────────────
        # Go: day == N_WEEKLY_SCHOOL_DAYS-1 and hours > MAX/2 → -1.0, else → +1.0
        is_saturday = (day == N_WEEKLY_SCHOOL_DAYS - 1)
        if is_saturday and day_total_hours > (PREFERRED_MAX_CLASS_HOUR_PER_DAY / 2.0):
            week_sched_fitness -= 1.0
        else:
            week_sched_fitness += 1.0

    # empty week
    if days_with_class == 0.0:
        return -24.0

    # normalize by number of days that actually had classes
    week_sched_fitness = week_sched_fitness / days_with_class

    # ── Days-per-week reward / punishment ────────────────────────────────────
    # Go: days_with_class > 4 → -2.0, else → +2.5
    if days_with_class > 4:
        week_sched_fitness -= 2.0
    else:
        week_sched_fitness += 2.5

    return week_sched_fitness


def measure_uni_sched_basic_fitness(
    complete_uni_sched: Dict[str, Any],
    department_to_measure: List[int] = None,
) -> float:
    """
    Matches Go: MeasureUniSchedBasicFitness()

    Iterates over every section's WeekTimeTable, averages their fitness scores.
    If department_to_measure is provided (list of dept IDs), only those
    departments are included — matching the Go departmentToMeasure map behaviour.

    Args:
        complete_uni_sched: dict keyed by section identifier, each value contains
                            'week_schedule' and optionally 'department_id'.
        department_to_measure: optional list of department IDs to filter on.

    Returns:
        float fitness score, or -24.0 if empty.
    """
    if not complete_uni_sched:
        return -24.0

    accumulated_fitness      = 0.0
    total_fitness_measurements = 0

    for section_key, section_data in complete_uni_sched.items():
        # department filter (mirrors Go: if !departmentToMeasure[deptID] → skip)
        if department_to_measure:
            dept_id = section_data.get('department_id')
            if dept_id not in department_to_measure:
                continue

        week_sched = section_data.get('week_schedule')
        if week_sched is None:
            continue

        total_fitness_measurements += 1
        accumulated_fitness += measure_week_timetable_basic_fitness(week_sched)

    if total_fitness_measurements == 0:
        return -24.0

    return accumulated_fitness / float(total_fitness_measurements)


# ── File I/O helpers ───────────────────────────────────────────────────────────

def load_schedules_json(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        raw_data = f.read().strip()

    if not raw_data:
        raise ValueError(f"JSON file is empty: {file_path}")

    data = json.loads(raw_data)

    if isinstance(data, list):
        return data
    elif 'schedules' in data:
        return data['schedules']
    elif 'schedule' in data:
        return [data]
    else:
        raise ValueError("Unknown JSON format. Expected list or dict with 'schedules' key.")


# ── Main entry point ───────────────────────────────────────────────────────────

def calculate_and_save(input_file: str, output_file: str = None, verbose: bool = True):
    print(f"\n📂 Loading schedules from: {input_file}")

    schedules = load_schedules_json(input_file)
    print(f"✓ Loaded {len(schedules)} schedule(s)")
    print("\n🧮 Calculating fitness scores...")

    results = []

    for idx, schedule_data in enumerate(schedules):
        # ── locate week_schedule ──────────────────────────────────────────────
        if 'week_schedule' in schedule_data:
            week_schedule = schedule_data['week_schedule']
        elif 'schedule' in schedule_data and 'week_schedule' in schedule_data['schedule']:
            week_schedule = schedule_data['schedule']['week_schedule']
        else:
            print(f"   ⚠️  Schedule {idx}: No week_schedule found, skipping")
            continue

        # ── validate dimensions ───────────────────────────────────────────────
        if len(week_schedule) != N_WEEKLY_SCHOOL_DAYS:
            print(f"   ⚠️  Schedule {idx}: Expected {N_WEEKLY_SCHOOL_DAYS} days, got {len(week_schedule)}, skipping")
            continue

        # ── calculate fitness (single section WeekTimeTable path) ─────────────
        fitness = measure_week_timetable_basic_fitness(week_schedule)

        result = {
            'id': schedule_data.get('id', f'schedule_{idx}'),
            'fitness': fitness,
            'week_schedule': week_schedule,
            'metadata': {
                'source': 'historical_manual_schedule',
                'calculated_by': 'calculate_fitness_for_historical.py',
            }
        }

        if 'metadata' in schedule_data:
            result['metadata'].update(schedule_data['metadata'])

        results.append(result)

        if verbose:
            print(f"   ✓ Schedule {idx} (ID: {result['id']}): Fitness = {fitness:.4f}")

    # ── statistics ────────────────────────────────────────────────────────────
    if results:
        fitnesses = [r['fitness'] for r in results]
        print(f"\n📊 Statistics:")
        print(f"   • Total schedules : {len(results)}")
        print(f"   • Fitness range   : {min(fitnesses):.4f} – {max(fitnesses):.4f}")
        print(f"   • Average fitness : {statistics.mean(fitnesses):.4f}")
        print(f"   • Median fitness  : {statistics.median(fitnesses):.4f}")
        print(f"   • Std deviation   : {statistics.pstdev(fitnesses):.4f}")

    # ── save ──────────────────────────────────────────────────────────────────
    if output_file is None:
        p = Path(input_file)
        output_file = str(p.parent / f"{p.stem}_with_fitness.json")

    output_data = {
        'schedules': results,
        'metadata': {
            'source_file': input_file,
            'total_schedules': len(results),
            'fitness_function': 'MeasureWeekTimeTableBasicFitness (Go backend parity)',
        }
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Saved {len(results)} schedules to: {output_file}")
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python calculate_fitness_for_historical.py <input_json> [output_json]")
        sys.exit(1)

    input_file  = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(input_file).exists():
        print(f"❌ Error: File not found: {input_file}")
        sys.exit(1)

    try:
        calculate_and_save(input_file, output_file, verbose=True)
        print("\n✅ Done.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()