"""
Example Go integration client for calling ANN API
Save this as: go_integration_client.go
Place it in your Go backend project
"""

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"time"
)

// ScheduleData represents schedule information
type ScheduleData struct {
	WeekSchedule [][][]int `json:"week_schedule"`
}

// FitnessRequest represents fitness prediction request
type FitnessRequest struct {
	Schedule ScheduleData `json:"schedule"`
}

// FitnessResponse represents fitness prediction response
type FitnessResponse struct {
	PredictedFitness float64 `json:"predicted_fitness"`
	Confidence       float64 `json:"confidence"`
	ProcessingTimeMs float64 `json:"processing_time_ms"`
}

// ConstraintResponse represents constraint checking response
type ConstraintResponse struct {
	Violations       map[string]bool    `json:"violations"`
	ViolationScores  map[string]float64 `json:"violation_scores"`
	ProcessingTimeMs float64            `json:"processing_time_ms"`
}

// ANNClient handles communication with ANN API
type ANNClient struct {
	BaseURL    string
	HTTPClient *http.Client
}

// NewANNClient creates a new ANN API client
func NewANNClient(baseURL string) *ANNClient {
	return &ANNClient{
		BaseURL: baseURL,
		HTTPClient: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

// PredictFitness calls the fitness prediction API
func (client *ANNClient) PredictFitness(weekSchedule [][][]int) (float64, error) {
	request := FitnessRequest{
		Schedule: ScheduleData{
			WeekSchedule: weekSchedule,
		},
	}

	requestBody, err := json.Marshal(request)
	if err != nil {
		return 0, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := client.HTTPClient.Post(
		client.BaseURL+"/predict/fitness",
		"application/json",
		bytes.NewBuffer(requestBody),
	)
	if err != nil {
		return 0, fmt.Errorf("API call failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := ioutil.ReadAll(resp.Body)
		return 0, fmt.Errorf("API returned error: %s", string(body))
	}

	var response FitnessResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return 0, fmt.Errorf("failed to decode response: %w", err)
	}

	return response.PredictedFitness, nil
}

// CheckConstraints calls the constraint checking API
func (client *ANNClient) CheckConstraints(weekSchedule [][][]int) (*ConstraintResponse, error) {
	request := FitnessRequest{
		Schedule: ScheduleData{
			WeekSchedule: weekSchedule,
		},
	}

	requestBody, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := client.HTTPClient.Post(
		client.BaseURL+"/predict/constraints",
		"application/json",
		bytes.NewBuffer(requestBody),
	)
	if err != nil {
		return nil, fmt.Errorf("API call failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := ioutil.ReadAll(resp.Body)
		return nil, fmt.Errorf("API returned error: %s", string(body))
	}

	var response ConstraintResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &response, nil
}

// HealthCheck checks if the ANN API is available
func (client *ANNClient) HealthCheck() error {
	resp, err := client.HTTPClient.Get(client.BaseURL + "/health")
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("API is unhealthy: status %d", resp.StatusCode)
	}

	return nil
}

// BatchPredictFitness predicts fitness for multiple schedules efficiently
func (client *ANNClient) BatchPredictFitness(schedules [][][][]int) ([]float64, error) {
	results := make([]float64, len(schedules))
	errors := make(chan error, len(schedules))
	resultsChan := make(chan struct {
		index   int
		fitness float64
	}, len(schedules))

	// Concurrent predictions
	for i, schedule := range schedules {
		go func(idx int, sched [][][]int) {
			fitness, err := client.PredictFitness(sched)
			if err != nil {
				errors <- err
				return
			}
			resultsChan <- struct {
				index   int
				fitness float64
			}{idx, fitness}
		}(i, schedule)
	}

	// Collect results
	for i := 0; i < len(schedules); i++ {
		select {
		case result := <-resultsChan:
			results[result.index] = result.fitness
		case err := <-errors:
			return nil, err
		}
	}

	return results, nil
}

// Example usage in your GA code
func ExampleIntegration() {
	// Initialize ANN client
	annClient := NewANNClient("http://localhost:8000")

	// Check if ANN service is available
	if err := annClient.HealthCheck(); err != nil {
		fmt.Printf("ANN service not available: %v\n", err)
		fmt.Println("Falling back to standard fitness function...")
		return
	}

	// Example schedule (6 days x 24 time slots x 3 attributes)
	exampleSchedule := make([][][]int, 6)
	for day := 0; day < 6; day++ {
		exampleSchedule[day] = make([][]int, 24)
		for slot := 0; slot < 24; slot++ {
			exampleSchedule[day][slot] = []int{0, 0, 0} // subject, instructor, room
		}
	}

	// Predict fitness using ANN
	predictedFitness, err := annClient.PredictFitness(exampleSchedule)
	if err != nil {
		fmt.Printf("Fitness prediction failed: %v\n", err)
		return
	}

	fmt.Printf("Predicted fitness: %.2f\n", predictedFitness)

	// Check constraints
	constraints, err := annClient.CheckConstraints(exampleSchedule)
	if err != nil {
		fmt.Printf("Constraint checking failed: %v\n", err)
		return
	}

	fmt.Println("\nConstraint violations:")
	for constraint, violated := range constraints.Violations {
		if violated {
			score := constraints.ViolationScores[constraint]
			fmt.Printf("  ⚠️  %s (confidence: %.2f%%)\n", constraint, score*100)
		}
	}
}

// Integration into existing GA code
// Add this to your GeneticAlgorithm.go file:

/*
// Initialize ANN client at the start of RunGeneticAlgorithm
var annClient *ANNClient
if useANN {
    annClient = NewANNClient("http://localhost:8000")
    if err := annClient.HealthCheck(); err != nil {
        log.Printf("ANN service not available: %v. Using standard fitness.", err)
        annClient = nil
    } else {
        log.Println("ANN service connected successfully")
    }
}

// In your fitness evaluation loop, use ANN when available:
func evaluateFitness(schedule Schedule.UniTimeTables, annClient *ANNClient) float64 {
    if annClient != nil {
        // Try ANN prediction first (faster)
        weekSchedule := convertToArray(schedule)
        fitness, err := annClient.PredictFitness(weekSchedule)
        if err == nil {
            return fitness
        }
        log.Printf("ANN prediction failed: %v. Falling back to standard.", err)
    }
    
    // Fall back to standard fitness function
    return MeasureUniSchedBasicFitness(schedule, curriculums, department_to_encode, selected_semester)
}

// For population evaluation, use batch prediction:
func evaluatePopulationFitness(population []Schedule.UniTimeTables, annClient *ANNClient) []float64 {
    if annClient != nil {
        schedules := make([][][][]int, len(population))
        for i, sched := range population {
            schedules[i] = convertToArray(sched)
        }
        
        fitnesses, err := annClient.BatchPredictFitness(schedules)
        if err == nil {
            return fitnesses
        }
        log.Printf("Batch prediction failed: %v. Using standard fitness.", err)
    }
    
    // Fall back to standard evaluation
    fitnesses := make([]float64, len(population))
    for i, sched := range population {
        fitnesses[i] = MeasureUniSchedBasicFitness(sched, ...)
    }
    return fitnesses
}
*/

func main() {
	fmt.Println("ANN Integration Client Example")
	fmt.Println("===============================")
	ExampleIntegration()
}
