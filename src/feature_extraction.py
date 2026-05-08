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
    """Feature extractor used by the constraint classifier."""

    def __init__(self):
        self._extractor = ScheduleFeatureExtractor()

    def extract(self, schedule_data: Dict) -> np.ndarray:
        return self._extractor.extract_features(schedule_data)


def create_feature_extractors():
    """Factory used by the API service to initialize all extractors."""
    return {
        'fitness': FitnessFeatureExtractor(),
        'constraint': ConstraintFeatureExtractor(),
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