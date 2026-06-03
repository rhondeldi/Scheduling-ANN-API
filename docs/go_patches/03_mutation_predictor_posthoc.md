# Go Patch 3 - mutation predictor as post-hoc scorer (Option B)

**Repo:** `scheduling-system-backend`
**Files:**

- `GeneticAlgorithm/ANNClient.go`
- `GeneticAlgorithm/GeneticAlgorithm.go`

**Rebuild required:** yes
**Regenerate training data:** no
**Python change required:** `/predict/mutation` now expects before+after payload

## Why this patch exists

The mutation model was retrained on delta features computed from
(before schedule, after schedule, mutation type, before fitness, after fitness).
To use it correctly, the Go client must send the same inputs.

This patch rewires the Go client and the GA call site so the endpoint
receives a true post-hoc mutation record instead of a hypothetical one.

## Dependency

Apply Patch 1 (per-operator mutation logging) or otherwise track the
actual mutation operator. The model expects one of:
`day_swap_timeslots`, `subject_day_swap`, `slot_nudge`, `slot_day_nudge`.

## 1) Update the request/response types

In `ANNClient.go`, replace the existing mutation structs with:

```go
// mutationRequest matches the Python /predict/mutation contract.
type mutationRequest struct {
    BeforeSchedule SchedulePayload `json:"before_schedule"`
    AfterSchedule  SchedulePayload `json:"after_schedule"`
    MutationType   string          `json:"mutation_type"`
    BeforeFitness  float64         `json:"before_fitness"`
    AfterFitness   float64         `json:"after_fitness"`
}

// MutationPrediction is the response from /predict/mutation.
type MutationPrediction struct {
    Prediction       string             `json:"prediction"`
    Confidence       float64            `json:"confidence"`
    Probabilities    map[string]float64 `json:"probabilities"`
    ProcessingTimeMs float64            `json:"processing_time_ms"`
}
```

## 2) Update PredictMutation signature and body

Change the method signature to accept both schedules and fitness values:

```go
func (client *ANNClient) PredictMutation(
    beforeSchedule SchedulePayload,
    afterSchedule SchedulePayload,
    mutationType string,
    beforeFitness float64,
    afterFitness float64,
) (*MutationPrediction, error) {
    req := mutationRequest{
        BeforeSchedule: beforeSchedule,
        AfterSchedule:  afterSchedule,
        MutationType:   mutationType,
        BeforeFitness:  beforeFitness,
        AfterFitness:   afterFitness,
    }
    // POST /predict/mutation using req
}
```

## 3) Update the GA call site

In `GeneticAlgorithm.go`, replace the old call that passes only the
post-mutation schedule and a hardcoded type.

If you have Patch 1 applied (per-operator loop), call the predictor
inside that loop using the operator name.

Pseudo-outline:

```go
// beforeStates captured per operator (from Patch 1)
beforeScheduleData, ok := firstSectionScheduleData(
    prev_uni_sched, curriculums, department_to_encode, selected_semester,
)
// after schedule is the current mutated schedule
postScheduleData, ok2 := firstSectionScheduleData(
    population[i].UniSched, curriculums, department_to_encode, selected_semester,
)

if ok && ok2 {
    mutPrediction, err := ann_client.PredictMutation(
        beforeScheduleData,
        postScheduleData,
        op.name, // use actual operator
        preMutationFitness,
        postMutationFitness,
    )
    // keep existing revert rule:
    // revert if Prediction == "worsen" and Confidence >= threshold
}
```

Notes:

- `preMutationFitness` should be the fitness of the schedule snapshot
  before the operator ran.
- `postMutationFitness` should be the fitness after the operator.
- If you only have a single mutation per generation, you can call once
  after mutation, but the mutation_type must match the operator used.

## 4) Keep the existing revert logic

The decision rule can remain unchanged:

- revert when Prediction == "worsen"
- and Confidence >= annMutationRevertConfidence

## 5) Verify

1. `go build ./...`
2. Run a short GA session with ANN enabled.
3. Confirm:
   - /predict/mutation is called with before+after payload
   - predictions are non-empty and confidence values look sensible
   - revert counts remain in a plausible range

## Expected outcome

- Mutation predictor now sees the same feature semantics as training.
- Revert decisions are aligned with the model's intent.
- The endpoint no longer receives garbage inputs.

## Rollback

Restore the old PredictMutation signature and call site if you decide
to switch to Option A (fitness-delta revert).
