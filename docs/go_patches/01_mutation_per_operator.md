# Go Patch 1 — per-operator mutation logging, drop no-ops

**Repo:** `scheduling-system-backend`
**File:** `GeneticAlgorithm/GeneticAlgorithm.go`
**Rebuild required:** yes
**Regenerate `mutation_samples.jsonl` after applying:** yes

## What this fixes

The current logging emits one mutation row **per section per generation**, even
when no operator touched that section. ~90% of rows are byte-identical
before/after no-ops. The `mutation_type` field is hardcoded to `"combined"`,
which kills 4 of the 41 input features.

The patch:

1. Decomposes the four `ApplyRandom*` operators into a per-operator loop.
2. Snapshots state **before each operator** so the delta is attributable.
3. **Skips emission for any section where before == after** (no-op filter).
4. Labels each row with the actual operator name from the spec set:
   `day_swap_timeslots`, `subject_day_swap`, `slot_nudge`, `slot_day_nudge`.

Behavior preserved: all four operators still fire per individual per generation.
Only the logging granularity changes.

## Apply

Find the block starting around line 1290 (the `prev_uni_sched` copy) and ending
around line 1419 (the closing of the post-mutation logging loop). Replace
**lines 1317 through 1419** (everything from `// POINT B (pre)` through the end
of the `LogMutation` call) with the block below.

The constraint-logging block (`POINT B'`) above this section stays put — only
the mutation snapshot/apply/log changes.

```go
// POINT B — per-operator mutation sampling.  Snapshot each section's state
// just before each operator runs so the resulting (before, after, delta)
// triple is correctly attributed to a single operator from the spec set.
// Rows where the operator left the section unchanged (byte-identical
// before/after) are skipped — those carry no learning signal and would
// otherwise dominate the dataset 9:1.
type beforeState struct {
    slice   [][][]int
    fitness float64
}

snapshot := func() map[int]beforeState {
    states := make(map[int]beforeState)
    IterateSectionsWeekSchedule(population[i].UniSched, curriculums, selected_semester, nil, nil,
        func(indices IterIndices, values IterValues) IterReturnType {
            if values.WeekSched == nil || values.Curriculum == nil {
                return IterProceed
            }
            if !department_to_encode[values.Curriculum.DepartmentID] {
                return IterProceed
            }
            states[indices.Usi] = beforeState{
                slice:   weekTimeTableToSlice(*values.WeekSched),
                fitness: MeasureWeekTimeTableBasicFitness(*values.WeekSched),
            }
            return IterProceed
        })
    return states
}

type operatorApplication struct {
    name string
    fn   func()
}
operators := []operatorApplication{
    {"day_swap_timeslots", func() {
        ApplyRandomDaySwapTimeSlots(population[i].UniSched, population[i].Resources, curriculums, department_id, selected_semester)
    }},
    {"subject_day_swap", func() {
        ApplyRandomSubjectDaySwap(population[i].UniSched, population[i].Resources, curriculums, department_id, selected_semester)
    }},
    {"slot_nudge", func() {
        ApplyRandomSubjectTimeSlotNudge(population[i].UniSched, population[i].Resources, curriculums, department_id, selected_semester)
    }},
    {"slot_day_nudge", func() {
        ApplyRandomSubjectTimeSlotAndDayNudge(population[i].UniSched, population[i].Resources, curriculums, department_id, selected_semester)
    }},
}

for _, op := range operators {
    var beforeStates map[int]beforeState
    if dataCollector.IsEnabled() {
        beforeStates = snapshot()
    }

    op.fn()

    if dataCollector.IsEnabled() && beforeStates != nil {
        IterateSectionsWeekSchedule(population[i].UniSched, curriculums, selected_semester, nil, nil,
            func(indices IterIndices, values IterValues) IterReturnType {
                if values.WeekSched == nil || values.Curriculum == nil {
                    return IterProceed
                }
                if !department_to_encode[values.Curriculum.DepartmentID] {
                    return IterProceed
                }
                before, exists := beforeStates[indices.Usi]
                if !exists {
                    return IterProceed
                }
                afterSlice := weekTimeTableToSlice(*values.WeekSched)
                if scheduleSlicesEqual(before.slice, afterSlice) {
                    return IterProceed // no-op for this section under this operator
                }
                afterFitness := MeasureWeekTimeTableBasicFitness(*values.WeekSched)
                delta := afterFitness - before.fitness
                label := "neutral"
                if delta > 0.5 {
                    label = "improve"
                } else if delta < -0.5 {
                    label = "worsen"
                }
                dataCollector.LogMutation(MutationSample{
                    BeforeSchedule: before.slice,
                    AfterSchedule:  afterSlice,
                    MutationType:   op.name,
                    BeforeFitness:  before.fitness,
                    AfterFitness:   afterFitness,
                    Delta:          delta,
                    Label:          label,
                    DepartmentID:   department_id,
                    Generation:     g,
                })
                return IterProceed
            },
        )
    }
}
```

## Helper to add (same file or `data_collector.go` — your call)

```go
// scheduleSlicesEqual returns true when two [6][24][3] schedule slices
// match cell-for-cell.  Used by the mutation logger to drop no-op rows
// where the operator didn't touch a given section.
func scheduleSlicesEqual(a, b [][][]int) bool {
    if len(a) != len(b) {
        return false
    }
    for d := range a {
        if len(a[d]) != len(b[d]) {
            return false
        }
        for s := range a[d] {
            if len(a[d][s]) != len(b[d][s]) {
                return false
            }
            for k := range a[d][s] {
                if a[d][s][k] != b[d][s][k] {
                    return false
                }
            }
        }
    }
    return true
}
```

## Expected outcome after regenerating `mutation_samples.jsonl`

Re-run `python scripts/verify_training_data.py --kind mutation`. You should see:

- `mutation_types_total` ≈ same order of magnitude as before, but only **changed**
  sections per operator are logged (much smaller file).
- `mutation_types_in_spec` should equal `mutation_types_total` (no more `"combined"`).
- Class imbalance drops from **41.8 : 1** to something in the **2:1 — 5:1** range,
  depending on how often each operator improves vs. worsens vs. produces a
  small-delta neutral.
- The trainer's mutation-type one-hot feature block becomes non-zero for every row.

If after regenerating, `mutation_types_total` is **much** smaller than expected
(say <5% of the original count), check that the operators are actually
modifying sections — a 100% no-op rate would indicate an operator bug rather
than a data bug.
