# Go Patch 2 — attach precomputed cross-section conflict aggregates

**Repo:** `scheduling-system-backend`
**Files:**
- `GeneticAlgorithm/data_collector.go`
- `GeneticAlgorithm/GeneticAlgorithm.go`

**Rebuild required:** yes
**Regenerate `constraint_samples.jsonl` after applying:** yes
**Replaces:** the Python augmentation stop-gap
(`scripts/augment_constraint_positives.py`) once this lands and the GA produces
real conflict positives.

## What this fixes

The current `ConstraintSample` only carries the target section's grid and a
violations dict. The trainer's feature extractor needs cross-section
conflict counts (instructor/room collisions across sections at the same
day+slot), and computing them in Python from a full `full_uni_schedule`
attached to every row would blow up disk by ~50×.

This patch precomputes 5 aggregate counts on the Go side once per individual
and attaches them to each section's sample. Tiny on disk, accurate, and the
Python feature extractor already knows how to consume them.

## 1. Extend the sample schema

In `GeneticAlgorithm/data_collector.go`, just below the existing
`ConstraintLabels`/`ConstraintSample` types, add:

```go
// CrossSectionAggregates summarises instructor/room collisions between the
// target section's occupied slots and every other section's occupied slots
// at the same (day, slot).  Computed once per individual and attached to
// every section sample so the Python feature extractor doesn't need the
// full university schedule serialised in each row.
type CrossSectionAggregates struct {
    TotalInstructorConflicts          int `json:"total_instructor_conflicts"`
    TotalRoomConflicts                int `json:"total_room_conflicts"`
    MaxInstructorConflictsInOneSlot   int `json:"max_instructor_conflicts_in_one_slot"`
    MaxRoomConflictsInOneSlot         int `json:"max_room_conflicts_in_one_slot"`
    SlotsWithAnyConflict              int `json:"slots_with_any_conflict"`
}
```

Then extend `ConstraintSample`:

```go
type ConstraintSample struct {
    SectionSchedule [][][]int              `json:"section_schedule"`
    DepartmentID    uint16                 `json:"department_id"`
    Violations      ConstraintLabels       `json:"violations"`
    CrossSection    CrossSectionAggregates `json:"cross_section"`   // NEW
}
```

Add a helper that builds the per-slot maps once and computes per-section
aggregates:

```go
// crossSectionStats walks every section in `sched` once and returns, for
// each (day, slot), the multiset of instructor ids and room ids in use.
// Used by computeAggregatesForSection.
type crossSectionStats struct {
    instructorAt [Const.N_WEEKLY_SCHOOL_DAYS][Const.N_DAILY_TIME_SLOTS]map[uint16]int
    roomAt       [Const.N_WEEKLY_SCHOOL_DAYS][Const.N_DAILY_TIME_SLOTS]map[uint16]int
}

func newCrossSectionStats(
    sched Schedule.UniTimeTables,
    curriculums []Curriculum.Curriculum,
    departmentToMeasure map[uint16]bool,
    selectedSemester int,
) *crossSectionStats {
    cs := &crossSectionStats{}
    for d := 0; d < Const.N_WEEKLY_SCHOOL_DAYS; d++ {
        for s := 0; s < Const.N_DAILY_TIME_SLOTS; s++ {
            cs.instructorAt[d][s] = map[uint16]int{}
            cs.roomAt[d][s] = map[uint16]int{}
        }
    }
    IterateSectionsWeekSchedule(sched, curriculums, selectedSemester, nil, nil,
        func(indices IterIndices, values IterValues) IterReturnType {
            if values.WeekSched == nil || values.Curriculum == nil {
                return IterProceed
            }
            if len(departmentToMeasure) > 0 && !departmentToMeasure[values.Curriculum.DepartmentID] {
                return IterProceed
            }
            for d := 0; d < Const.N_WEEKLY_SCHOOL_DAYS; d++ {
                for s := 0; s < Const.N_DAILY_TIME_SLOTS; s++ {
                    ts := (*values.WeekSched)[d][s]
                    if ts.GetSubjectID() == 0 {
                        continue
                    }
                    if iid := ts.GetInstructorID(); iid != 0 {
                        cs.instructorAt[d][s][iid]++
                    }
                    if rid := ts.GetRoomID(); rid != 0 {
                        cs.roomAt[d][s][rid]++
                    }
                }
            }
            return IterProceed
        })
    return cs
}

// computeAggregatesForSection returns the 5 cross-section aggregates for one
// section against the pre-built per-slot multiset.  Counts only OTHER
// sections by subtracting the target's own occupancy from each count.
func (cs *crossSectionStats) computeAggregatesForSection(
    section Schedule.WeekTimeTable,
) CrossSectionAggregates {
    out := CrossSectionAggregates{}
    distinctConflictSlots := 0

    for d := 0; d < Const.N_WEEKLY_SCHOOL_DAYS; d++ {
        for s := 0; s < Const.N_DAILY_TIME_SLOTS; s++ {
            ts := section[d][s]
            if ts.GetSubjectID() == 0 {
                continue
            }
            iid := ts.GetInstructorID()
            rid := ts.GetRoomID()
            instConflicts := 0
            roomConflicts := 0
            if iid != 0 {
                // total - self (1) = number of OTHER sections using this instructor here
                if total := cs.instructorAt[d][s][iid]; total > 0 {
                    instConflicts = total - 1
                }
            }
            if rid != 0 {
                if total := cs.roomAt[d][s][rid]; total > 0 {
                    roomConflicts = total - 1
                }
            }
            out.TotalInstructorConflicts += instConflicts
            out.TotalRoomConflicts += roomConflicts
            if instConflicts > out.MaxInstructorConflictsInOneSlot {
                out.MaxInstructorConflictsInOneSlot = instConflicts
            }
            if roomConflicts > out.MaxRoomConflictsInOneSlot {
                out.MaxRoomConflictsInOneSlot = roomConflicts
            }
            if instConflicts > 0 || roomConflicts > 0 {
                distinctConflictSlots++
            }
        }
    }
    out.SlotsWithAnyConflict = distinctConflictSlots
    return out
}
```

## 2. Update the three `LogConstraint` call sites

Each call site in `GeneticAlgorithm/GeneticAlgorithm.go` follows the same
pattern. **Build the stats once per individual, then attach the per-section
aggregate to each `ConstraintSample`.**

The current call sites are around:
- **Genesis** (~line 893–913): inside the genesis loop, after
  `detectHardViolations`.
- **Post-mutation** (~line 1354–1376): inside the per-individual mutation
  block, after the operator decomposition from patch 1.
- **Post-evolution snapshot** (~line 1368): wherever the third
  `LogConstraint` is invoked.

For each one, change:

```go
hardViolations := detectHardViolations(initial_sched, curriculums, rooms,
    department_to_encode, selected_semester)
IterateSectionsWeekSchedule(initial_sched, curriculums, selected_semester, nil, nil,
    func(indices IterIndices, values IterValues) IterReturnType {
        ...
        soft := detectSoftViolationsForSection(*values.WeekSched)
        dataCollector.LogConstraint(ConstraintSample{
            SectionSchedule: weekTimeTableToSlice(*values.WeekSched),
            DepartmentID:    department_id,
            Violations:      mergeViolations(hardViolations, soft),
        })
        return IterProceed
    },
)
```

into:

```go
hardViolations := detectHardViolations(initial_sched, curriculums, rooms,
    department_to_encode, selected_semester)
crossStats := newCrossSectionStats(initial_sched, curriculums,
    department_to_encode, selected_semester)

IterateSectionsWeekSchedule(initial_sched, curriculums, selected_semester, nil, nil,
    func(indices IterIndices, values IterValues) IterReturnType {
        ...
        soft := detectSoftViolationsForSection(*values.WeekSched)
        dataCollector.LogConstraint(ConstraintSample{
            SectionSchedule: weekTimeTableToSlice(*values.WeekSched),
            DepartmentID:    department_id,
            Violations:      mergeViolations(hardViolations, soft),
            CrossSection:    crossStats.computeAggregatesForSection(*values.WeekSched),
        })
        return IterProceed
    },
)
```

Repeat for the other two call sites (use the appropriate `sched` variable —
`population[i].UniSched` in the mutation block, etc.).

## Expected outcome after regenerating `constraint_samples.jsonl`

Re-run `python scripts/verify_training_data.py --kind constraint`.

- `with_cross_section_aggregates` should equal the total record count.
- `instructor_conflict` / `room_conflict` will start with **0 positives** at
  first — because the GA's operators preserve feasibility within a single
  individual. To get real positives you need to either:
  - Sample mutated *intermediate* states (before the rollback), or
  - Keep using the Python `augment_constraint_positives.py` script — but now
    it can leverage real cross-section counts for negatives instead of just
    zeros, which makes the model's decision boundary more meaningful.

The cross-section feature columns will be populated with the **real**
inter-section interference numbers for every section — meaning even
negative examples carry signal (e.g. high `total_room_conflicts` with
`room_conflict=false` would be weird and the model can learn the boundary).

## After this lands

Once you have real-data positives flowing, remove the dependency on the
Python augmentation script by training directly against
`data/training_output/constraint_samples.jsonl` instead of the
`.augmented.jsonl` variant.
