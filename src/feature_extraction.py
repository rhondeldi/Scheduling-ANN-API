"""
Feature extraction from schedule data for ANN fitness predictor.

All slot indices and thresholds mirror the Go backend exactly:
  - Slot 0  = 7:00 AM
  - Slot 8  = 11:00 AM  (lunch window start)
  - Slot 11 = 12:30 PM  (lunch window end, inclusive)
  - Slot 20 = 5:00 PM   (late-class threshold)
  - Each slot = 30 min  (N_HOUR_TIME_SLOTS = 2)

Total features: 48  (matches config.FITNESS_PREDICTOR_CONFIG['input_dim'])
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Any

from . import config


# ── constants (mirrors Go backend) ────────────────────────────────────────────
LUNCH_START  = 8    # slot 8  = 11:00 AM
LUNCH_END    = 11   # slot 11 = 12:30 PM  (inclusive, matches Go: ts <= 11)
LATE_SLOT    = 20   # slot 20 = 5:00 PM
SAT_DAY      = config.N_WEEKLY_SCHOOL_DAYS - 1   # index 5
PREFERRED_MAX_HOURS = 10.0


class ScheduleFeatureExtractor:
    """
    Extract 48 features from a week_schedule for the fitness predictor ANN.

    Feature groups
    ──────────────
    G1  Temporal distribution   12 features
    G2  Constraint indicators   12 features
    G3  Resource utilisation     7 features
    G4  Distribution quality     9 features
    G5  Workload balance         8 features
    ─────────────────────────────────────────
    Total                       48 features
    """

    def __init__(self):
        self.n_days  = config.N_WEEKLY_SCHOOL_DAYS
        self.n_slots = config.N_DAILY_TIME_SLOTS

    # ── public API ─────────────────────────────────────────────────────────────

    def extract_features(self, schedule_data: Dict[str, Any]) -> np.ndarray:
        schedule = self._parse_schedule(schedule_data)

        features: List[float] = []
        features.extend(self._temporal(schedule))       # G1 — 12
        features.extend(self._constraints(schedule))    # G2 — 12
        features.extend(self._resources(schedule))      # G3 —  7
        features.extend(self._distribution(schedule))   # G4 —  9
        features.extend(self._workload(schedule))       # G5 —  8

        arr = np.array(features, dtype=np.float32)
        assert arr.shape[0] == 48, f"Feature count mismatch: {arr.shape[0]} != 48"
        return arr

    # ── parsing ────────────────────────────────────────────────────────────────

    def _parse_schedule(self, schedule_data: Any) -> np.ndarray:
        """Return (n_days, n_slots, 3) int32 array."""
        if isinstance(schedule_data, np.ndarray) and schedule_data.ndim == 3:
            return schedule_data.astype(np.int32)

        raw_ws = None
        if isinstance(schedule_data, dict):
            raw_ws = schedule_data.get("week_schedule")

        parsed = np.zeros((self.n_days, self.n_slots, 3), dtype=np.int32)
        if not isinstance(raw_ws, list):
            return parsed

        for d, day in enumerate(raw_ws[:self.n_days]):
            if not isinstance(day, list):
                continue
            for s, slot in enumerate(day[:self.n_slots]):
                if isinstance(slot, (list, tuple)) and len(slot) >= 3:
                    try:
                        parsed[d, s] = [int(slot[0]), int(slot[1]), int(slot[2])]
                    except Exception:
                        pass
        return parsed

    # ── G1: Temporal (12) ──────────────────────────────────────────────────────

    def _temporal(self, sched: np.ndarray) -> List[float]:
        """
        [0-5]  daily_hours[0..5]
        [6]    total_weekly_hours
        [7]    days_with_classes
        [8]    hour_variance
        [9]    avg_hours_per_active_day
        [10]   saturday_hours
        [11]   max_daily_hours
        """
        daily_hours = [
            float(np.sum(sched[d, :, 0] > 0)) / config.N_HOUR_TIME_SLOTS
            for d in range(self.n_days)
        ]
        total        = sum(daily_hours)
        active_days  = sum(1 for h in daily_hours if h > 0)
        variance     = float(np.var(daily_hours))
        avg          = total / active_days if active_days > 0 else 0.0
        saturday     = daily_hours[SAT_DAY]
        max_day      = max(daily_hours) if daily_hours else 0.0

        return daily_hours + [total, float(active_days), variance, avg, saturday, max_day]

    # ── G2: Constraints (12) ───────────────────────────────────────────────────

    def _constraints(self, sched: np.ndarray) -> List[float]:
        """
        Per day (6 days × 2 = 12):
          [even]  has_lunch_window_free  — 1 if ANY slot in [8,11] is empty
                  (mirrors Go: ts >= 8 && ts <= 11 && subject == 0)
          [odd]   late_class_count       — occupied slots at index >= 20
        """
        feats: List[float] = []
        for d in range(self.n_days):
            subj = sched[d, :, 0]

            # lunch: at least one free slot in the window [LUNCH_START, LUNCH_END]
            has_lunch = float(np.any(subj[LUNCH_START: LUNCH_END + 1] == 0))

            # late classes: occupied slots from slot 20 onwards
            late_count = float(np.sum(subj[LATE_SLOT:] > 0))

            feats.extend([has_lunch, late_count])
        return feats

    # ── G3: Resources (7) ──────────────────────────────────────────────────────

    def _resources(self, sched: np.ndarray) -> List[float]:
        """
        [0]  unique_instructors
        [1]  unique_rooms
        [2]  unique_subjects
        [3]  instructor_load_variance
        [4]  instructor_max_load
        [5]  instructor_min_load
        [6]  instructor_avg_load
        """
        instr_ids = sched[:, :, 1].flatten()
        room_ids  = sched[:, :, 2].flatten()
        subj_ids  = sched[:, :, 0].flatten()

        unique_instr = float(len(np.unique(instr_ids[instr_ids > 0])))
        unique_rooms = float(len(np.unique(room_ids[room_ids  > 0])))
        unique_subj  = float(len(np.unique(subj_ids[subj_ids  > 0])))

        counts = {}
        for iid in instr_ids:
            if iid > 0:
                counts[iid] = counts.get(iid, 0) + 1

        if counts:
            vals = list(counts.values())
            var_ = float(np.var(vals))
            mx   = float(max(vals))
            mn   = float(min(vals))
            avg  = float(np.mean(vals))
        else:
            var_ = mx = mn = avg = 0.0

        return [unique_instr, unique_rooms, unique_subj, var_, mx, mn, avg]

    # ── G4: Distribution (9) ───────────────────────────────────────────────────

    def _distribution(self, sched: np.ndarray) -> List[float]:
        """
        [0-5]  daily_gaps[0..5]
        [6]    total_gaps
        [7]    avg_gaps_per_day
        [8]    max_continuous_teaching_slots
        """
        daily_gaps: List[float] = []
        max_cont = 0

        for d in range(self.n_days):
            occ = (sched[d, :, 0] > 0).astype(int)
            if occ.sum() == 0:
                daily_gaps.append(0.0)
            else:
                first = int(np.argmax(occ))
                last  = int(len(occ) - np.argmax(occ[::-1]) - 1)
                gaps  = (last - first + 1) - int(occ.sum())
                daily_gaps.append(float(gaps))

            # max continuous block
            cur = 0
            for v in sched[d, :, 0]:
                if v > 0:
                    cur += 1
                    max_cont = max(max_cont, cur)
                else:
                    cur = 0

        total_gaps = sum(daily_gaps)
        avg_gaps   = total_gaps / self.n_days

        return daily_gaps + [total_gaps, avg_gaps, float(max_cont)]

    # ── G5: Workload balance (8) ───────────────────────────────────────────────

    def _workload(self, sched: np.ndarray) -> List[float]:
        """
        [0]  morning_slots    (slots 0–7,   before 11 AM)
        [1]  afternoon_slots  (slots 8–19,  11 AM – 5 PM)
        [2]  evening_slots    (slots 20–23, after 5 PM)
        [3]  morning_ratio
        [4]  afternoon_ratio
        [5]  evening_ratio
        [6]  saturday_ratio   (saturday hours / total hours)
        [7]  spread_factor
        """
        morning   = float(np.sum(sched[:, :LUNCH_START, 0]     > 0))
        afternoon = float(np.sum(sched[:, LUNCH_START:LATE_SLOT, 0] > 0))
        evening   = float(np.sum(sched[:, LATE_SLOT:, 0]        > 0))
        total     = morning + afternoon + evening

        if total > 0:
            m_r = morning   / total
            a_r = afternoon / total
            e_r = evening   / total
        else:
            m_r = a_r = e_r = 0.0

        weekday_h  = float(np.sum(sched[:SAT_DAY, :, 0] > 0)) / config.N_HOUR_TIME_SLOTS
        saturday_h = float(np.sum(sched[SAT_DAY,  :, 0] > 0)) / config.N_HOUR_TIME_SLOTS
        sat_ratio  = saturday_h / (weekday_h + saturday_h) if (weekday_h + saturday_h) > 0 else 0.0

        occupied_idx = np.where(sched[:, :, 0].flatten() > 0)[0]
        if len(occupied_idx) > 1:
            spread = float((occupied_idx[-1] - occupied_idx[0]) / len(occupied_idx))
        else:
            spread = 0.0

        return [morning, afternoon, evening, m_r, a_r, e_r, sat_ratio, spread]


# ── convenience wrapper used by the trainer ───────────────────────────────────

class FitnessFeatureExtractor:
    """Thin wrapper kept for API compatibility."""

    def __init__(self):
        self._extractor = ScheduleFeatureExtractor()

    def extract(self, schedule_data: Dict) -> np.ndarray:
        return self._extractor.extract_features(schedule_data)


class ConstraintFeatureExtractor:
    """29-feature extractor for the constraint classifier.

    Layout
    ──────
    Group 1 — Per-section soft constraint features (24)
      For each of 6 days (4 × 6 = 24):
        [has_lunch_free, late_class_count, daily_hours, is_saturday_overload]

    Group 2 — Cross-section conflict aggregates (5)
        [total_instructor_conflicts,
         total_room_conflicts,
         max_instructor_conflicts_in_one_slot,
         max_room_conflicts_in_one_slot,
         slots_with_any_conflict]
    """

    EXPECTED_DIM = 29
    SAT_OVERLOAD_HOURS = 5.0

    def __init__(self):
        self.n_days = config.N_WEEKLY_SCHOOL_DAYS
        self.n_slots = config.N_DAILY_TIME_SLOTS
        self._parser = ScheduleFeatureExtractor()

    CROSS_AGGREGATE_KEYS = (
        "total_instructor_conflicts",
        "total_room_conflicts",
        "max_instructor_conflicts_in_one_slot",
        "max_room_conflicts_in_one_slot",
        "slots_with_any_conflict",
    )

    # ── public API ────────────────────────────────────────────────────────────
    def extract(
        self,
        schedule_data: Any,
        full_uni_schedule: Any = None,
        cross_aggregates: Any = None,
    ) -> np.ndarray:
        """Build the 29-feature vector.

        Args:
            schedule_data: the target section — either a dict
                {"week_schedule": [...]}, {"section_schedule": [...]},
                {"schedule": [...]}, or a raw [6][24][3] array.
            full_uni_schedule: optional iterable of other sections. Used
                only when `cross_aggregates` is not supplied.
            cross_aggregates: optional dict of precomputed cross-section
                conflict counts (the five fields in CROSS_AGGREGATE_KEYS).
                When present, takes precedence over `full_uni_schedule` —
                the Go data collector emits these directly to keep the
                JSONL compact.

        Cross-section features are zeroed when neither source is given.
        """
        section = self._coerce(schedule_data)

        soft = self._soft_features(section)
        if cross_aggregates is not None:
            cross = self._cross_from_aggregates(cross_aggregates)
        else:
            cross = self._cross_section_features(section, full_uni_schedule)

        arr = np.asarray(soft + cross, dtype=np.float32)
        assert arr.shape[0] == self.EXPECTED_DIM, (
            f"Constraint feature dim {arr.shape[0]} != {self.EXPECTED_DIM}"
        )
        return arr

    # ── precomputed aggregates path ───────────────────────────────────────────
    def _cross_from_aggregates(self, agg: Any) -> List[float]:
        if not isinstance(agg, dict):
            return [0.0] * 5
        out: List[float] = []
        for key in self.CROSS_AGGREGATE_KEYS:
            val = agg.get(key, 0)
            try:
                out.append(float(val))
            except (TypeError, ValueError):
                out.append(0.0)
        return out

    # ── helpers ───────────────────────────────────────────────────────────────
    def _coerce(self, data: Any) -> np.ndarray:
        """Normalize one section schedule to a (n_days, n_slots, 3) int array."""
        if isinstance(data, np.ndarray) and data.ndim == 3:
            return data.astype(np.int32)
        if isinstance(data, dict):
            for key in ("section_schedule", "week_schedule", "schedule"):
                if key in data and data[key] is not None:
                    return self._parser._parse_schedule({"week_schedule": data[key]})
            return self._parser._parse_schedule(data)
        if isinstance(data, list):
            return self._parser._parse_schedule({"week_schedule": data})
        return np.zeros((self.n_days, self.n_slots, 3), dtype=np.int32)

    def _soft_features(self, section: np.ndarray) -> List[float]:
        feats: List[float] = []
        for d in range(self.n_days):
            subj = section[d, :, 0]
            has_lunch = float(np.any(subj[LUNCH_START:LUNCH_END + 1] == 0))
            late_count = float(np.sum(subj[LATE_SLOT:] > 0))
            daily_hours = float(np.sum(subj > 0)) / config.N_HOUR_TIME_SLOTS
            is_sat_overload = float(
                d == SAT_DAY and daily_hours > self.SAT_OVERLOAD_HOURS
            )
            feats.extend([has_lunch, late_count, daily_hours, is_sat_overload])
        return feats

    def _cross_section_features(
        self,
        section: np.ndarray,
        full_uni_schedule: Any,
    ) -> List[float]:
        if not full_uni_schedule:
            return [0.0, 0.0, 0.0, 0.0, 0.0]

        # Coerce + filter out any section identical to the target (so we don't
        # count it as a conflict against itself).
        others: List[np.ndarray] = []
        for raw in full_uni_schedule:
            try:
                arr = self._coerce(raw)
            except Exception:
                continue
            if arr.shape != section.shape:
                continue
            if np.array_equal(arr, section):
                continue
            others.append(arr)

        if not others:
            return [0.0, 0.0, 0.0, 0.0, 0.0]

        stacked = np.stack(others, axis=0)  # (n_other, n_days, n_slots, 3)
        other_subj = stacked[..., 0]
        other_instr = stacked[..., 1]
        other_room = stacked[..., 2]
        # Only count slots where the other section is actually occupied.
        other_occupied = other_subj > 0

        sec_subj = section[..., 0]
        sec_instr = section[..., 1]
        sec_room = section[..., 2]
        target_occupied = sec_subj > 0  # (n_days, n_slots)

        # broadcast: compare each other section against target slot-by-slot
        instr_match = (other_instr == sec_instr[None, ...]) & (sec_instr[None, ...] > 0) & other_occupied
        room_match = (other_room == sec_room[None, ...]) & (sec_room[None, ...] > 0) & other_occupied

        # only meaningful where the target itself is occupied
        instr_match &= target_occupied[None, ...]
        room_match &= target_occupied[None, ...]

        # Per-slot counts (sum across other sections)
        per_slot_instr = instr_match.sum(axis=0)  # (n_days, n_slots)
        per_slot_room = room_match.sum(axis=0)

        total_instr = int(per_slot_instr.sum())
        total_room = int(per_slot_room.sum())
        max_instr = int(per_slot_instr.max(initial=0))
        max_room = int(per_slot_room.max(initial=0))
        slots_with_any = int(np.sum((per_slot_instr + per_slot_room) > 0))

        return [
            float(total_instr),
            float(total_room),
            float(max_instr),
            float(max_room),
            float(slots_with_any),
        ]


class CrossoverFeatureExtractor:
    """23-feature extractor for the crossover compatibility classifier.

    Layout
    ──────
      [parent1_fitness, parent2_fitness]                           (2)
      [fitness_diff (p1 - p2), fitness_avg]                        (2)
      [matching_slot_count, matching_ratio, occupied_slot_overlap] (3)
      Per parent (6 features each, 12 total):
        [days_with_class, total_hours, has_saturday_classes,
         lunch_break_days, late_class_count, days_over_preferred_hours]
      Structural deltas (p1 - p2):                                  (4)
        [days_with_class_diff, total_hours_diff,
         lunch_break_days_diff, late_class_diff]
    """

    EXPECTED_DIM = 23
    PREFERRED_MAX_HOURS = 10.0

    def __init__(self):
        self.n_days = config.N_WEEKLY_SCHOOL_DAYS
        self.n_slots = config.N_DAILY_TIME_SLOTS
        self._sat_day = SAT_DAY
        self._lunch_start = LUNCH_START
        self._lunch_end_excl = LUNCH_END + 1
        self._late_start = LATE_SLOT
        self._parser = ScheduleFeatureExtractor()

    # ── public API ────────────────────────────────────────────────────────────
    def extract(
        self,
        parent1: Any,
        parent2: Any,
        parent1_fitness: float,
        parent2_fitness: float,
    ) -> np.ndarray:
        p1 = self._coerce(parent1)
        p2 = self._coerce(parent2)
        p1f = float(parent1_fitness)
        p2f = float(parent2_fitness)

        feats: List[float] = [p1f, p2f, p1f - p2f, 0.5 * (p1f + p2f)]
        feats.extend(self._similarity_features(p1, p2))

        s1 = self._parent_structural(p1)
        s2 = self._parent_structural(p2)
        feats.extend(s1)
        feats.extend(s2)
        feats.extend([s1[0] - s2[0], s1[1] - s2[1], s1[3] - s2[3], s1[4] - s2[4]])

        arr = np.asarray(feats, dtype=np.float32)
        assert arr.shape[0] == self.EXPECTED_DIM, (
            f"Crossover feature dim {arr.shape[0]} != {self.EXPECTED_DIM}"
        )
        return arr

    # ── helpers ───────────────────────────────────────────────────────────────
    def _coerce(self, data: Any) -> np.ndarray:
        if isinstance(data, np.ndarray) and data.ndim == 3:
            return data.astype(np.int32)
        if isinstance(data, dict):
            for key in ("week_schedule", "section_schedule", "schedule"):
                if key in data and data[key] is not None:
                    return self._parser._parse_schedule({"week_schedule": data[key]})
            return self._parser._parse_schedule(data)
        if isinstance(data, list):
            return self._parser._parse_schedule({"week_schedule": data})
        return np.zeros((self.n_days, self.n_slots, 3), dtype=np.int32)

    def _similarity_features(self, p1: np.ndarray, p2: np.ndarray) -> List[float]:
        s1 = p1[:, :, 0]
        s2 = p2[:, :, 0]
        occ1 = s1 > 0
        occ2 = s2 > 0

        matching = int(np.sum((s1 == s2) & occ1 & occ2))
        overlap = int(np.sum(occ1 & occ2))
        union_occupied = int(np.sum(occ1 | occ2))

        ratio = (float(matching) / float(union_occupied)) if union_occupied > 0 else 0.0
        return [float(matching), ratio, float(overlap)]

    def _parent_structural(self, sched: np.ndarray) -> List[float]:
        subj = sched[:, :, 0]
        occ = subj > 0
        days_with_class = float(np.sum(np.any(occ, axis=1)))
        total_hours = float(np.sum(occ)) / float(config.N_HOUR_TIME_SLOTS)
        has_saturday = float(np.any(occ[self._sat_day]))

        lunch_break_days = 0
        for d in range(self.n_days):
            if np.any(subj[d, self._lunch_start:self._lunch_end_excl] == 0):
                lunch_break_days += 1

        late_class_count = float(np.sum(subj[:, self._late_start:] > 0))

        hours_per_day = np.sum(occ, axis=1) / float(config.N_HOUR_TIME_SLOTS)
        days_over_preferred = float(np.sum(hours_per_day > self.PREFERRED_MAX_HOURS))

        return [
            days_with_class,
            total_hours,
            has_saturday,
            float(lunch_break_days),
            late_class_count,
            days_over_preferred,
        ]


class MutationFeatureExtractor:
    """41-feature delta extractor for the mutation impact classifier."""

    EXPECTED_DIM = 41
    MUTATION_TYPES = [
        "day_swap_timeslots",
        "subject_day_swap",
        "slot_nudge",
        "slot_day_nudge",
    ]
    MUTATION_TYPE_TO_IDX = {t: i for i, t in enumerate(MUTATION_TYPES)}

    def __init__(self):
        self.n_days = config.N_WEEKLY_SCHOOL_DAYS
        self.n_slots = config.N_DAILY_TIME_SLOTS
        self._parser = ScheduleFeatureExtractor()

    def extract(
        self,
        before: Any,
        after: Any,
        mutation_type: str,
        before_fitness: float,
        after_fitness: float,
    ) -> np.ndarray:
        b = self._coerce(before)
        a = self._coerce(after)
        mtype = str(mutation_type or "").lower()

        feats: List[float] = []

        one_hot = [0.0] * len(self.MUTATION_TYPES)
        idx = self.MUTATION_TYPE_TO_IDX.get(mtype)
        if idx is not None:
            one_hot[idx] = 1.0
        feats.extend(one_hot)

        b_fit = float(before_fitness)
        a_fit = float(after_fitness)
        feats.extend([b_fit, a_fit, a_fit - b_fit])

        for d in range(self.n_days):
            feats.extend([
                self._has_lunch(b[d]),
                self._has_lunch(a[d]),
                self._has_late_class(b[d]),
                self._has_late_class(a[d]),
            ])

        for d in range(self.n_days):
            feats.append(self._daily_hours(a[d]) - self._daily_hours(b[d]))

        feats.extend([
            self._days_with_class(b),
            self._days_with_class(a),
        ])

        feats.extend([
            self._daily_hours(b[SAT_DAY]),
            self._daily_hours(a[SAT_DAY]),
        ])

        arr = np.asarray(feats, dtype=np.float32)
        assert arr.shape[0] == self.EXPECTED_DIM, (
            f"Mutation feature dim {arr.shape[0]} != {self.EXPECTED_DIM}"
        )
        return arr

    def _coerce(self, data: Any) -> np.ndarray:
        if isinstance(data, np.ndarray) and data.ndim == 3:
            return data.astype(np.int32)
        if isinstance(data, dict):
            for key in ("week_schedule", "section_schedule", "schedule"):
                if key in data and data[key] is not None:
                    return self._parser._parse_schedule({"week_schedule": data[key]})
            return self._parser._parse_schedule(data)
        if isinstance(data, list):
            return self._parser._parse_schedule({"week_schedule": data})
        return np.zeros((self.n_days, self.n_slots, 3), dtype=np.int32)

    def _has_lunch(self, day: np.ndarray) -> float:
        return float(np.any(day[LUNCH_START:LUNCH_END + 1, 0] == 0))

    def _has_late_class(self, day: np.ndarray) -> float:
        return float(np.any(day[LATE_SLOT:, 0] > 0))

    def _daily_hours(self, day: np.ndarray) -> float:
        return float(np.sum(day[:, 0] > 0)) / config.N_HOUR_TIME_SLOTS

    def _days_with_class(self, sched: np.ndarray) -> float:
        return float(np.sum(np.any(sched[:, :, 0] > 0, axis=1)))


def create_feature_extractors():
    """Factory used by the API service to initialize all extractors."""
    return {
        'fitness': FitnessFeatureExtractor(),
        'constraint': ConstraintFeatureExtractor(),
        'crossover': CrossoverFeatureExtractor(),
        'mutation': MutationFeatureExtractor(),
        'general': ScheduleFeatureExtractor(),
    }


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    dummy = {"week_schedule": np.random.randint(0, 10, (6, 24, 3)).tolist()}
    ext   = ScheduleFeatureExtractor()
    feats = ext.extract_features(dummy)
    print(f"Feature vector length : {len(feats)}  (expected 48)")
    print(f"dtype                 : {feats.dtype}")
    print(f"sample values         : {feats[:12]}")