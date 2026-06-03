# System Architecture - Hybrid ANN-Assisted Genetic Algorithm

**Last Updated**: May 16, 2026  
**Version**: 2.0.0

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Components](#system-components)
3. [Detailed Architecture](#detailed-architecture)
4. [Data Flow](#data-flow)
5. [Integration Points](#integration-points)
6. [Deployment Structure](#deployment-structure)
7. [Model Specifications](#model-specifications)
8. [API Endpoints](#api-endpoints)
9. [Performance Characteristics](#performance-characteristics)

---

## Overview

The system is a **hybrid scheduler** combining:

- **Go Backend**: Genetic Algorithm for schedule optimization
- **Python ANN API**: 4 neural network models for intelligent decision-making
- **Feature Engineering**: 48-feature extraction from schedule data
- **Data Pipeline**: Training data from manual/historical schedules

### Architecture Philosophy

```
Traditional GA             →    ANN-Assisted GA
(Slow fitness checks)           (Fast predictions)
  ├─ 1ms per evaluation    ├─ 0.01ms per evaluation
  └─ 100x slower           └─ 100x faster
```

---

## System Components

### 1. **Data Layer**

```
📁 Data Sources
  ├── data/
  │   ├── manual_schedules.json           (Historical valid schedules)
  │   ├── manual_schedules_with_fitness.json
  │   ├── subjects.json                   (Subject definitions)
  │   ├── instructors.json                (Instructor info)
  │   ├── rooms.json                      (Room definitions)
  │   └── training_output/                (Generated training data)
  │       ├── constraint_samples.jsonl
  │       ├── crossover_samples.jsonl
  │       ├── mutation_samples.jsonl
  │       └── training_data.json
  │
  └── 📁 logs/
      ├── OCR scanned schedules
      ├── Fitness check outputs
      └── GA run logs
```

**Data Characteristics**:

- **Manual Schedules**: ~50-100 historical reference schedules
- **Training Samples**: 1,000-5,000 synthetic variants per model
- **Format**: JSON with week-based schedule structure
- **Semester Metadata**: Derived from filename patterns (e.g., "2024_2025")

### 2. **Feature Extraction (Python)**

```
schedule_data (JSON)
       ↓
[ScheduleFeatureExtractor]
       ↓
48-dimensional feature vector
       ↓
[FeatureScaler: StandardScaler]
       ↓
Normalized features → Model Input
```

**Features Extracted** (48 total):

```
Group 1: Temporal Distribution (12 features)
  ├─ Early morning classes (7-9 AM)
  ├─ Mid-morning classes (9-11 AM)
  ├─ Lunch hour usage (11 AM-12:30 PM)
  ├─ Afternoon classes (12:30-3 PM)
  ├─ Late afternoon (3-5 PM)
  ├─ Evening classes (5+ PM)
  └─ Daily distribution metrics

Group 2: Constraint Indicators (12 features)
  ├─ Subject balance across days
  ├─ Instructor workload distribution
  ├─ Room utilization rates
  ├─ Duplicate subjects per day
  └─ Constraint violations (hard & soft)

Group 3: Resource Utilization (7 features)
  ├─ Room utilization percentage
  ├─ Instructor utilization
  ├─ Subject presence ratio
  └─ Time slot coverage

Group 4: Distribution Quality (9 features)
  ├─ Class spacing metrics
  ├─ Gap analysis
  └─ Spread uniformity

Group 5: Workload Balance (8 features)
  ├─ Instructor daily workload variance
  ├─ Subject hourly distribution
  └─ Load balancing metrics
```

**Code Location**: [src/feature_extraction.py](../src/feature_extraction.py)

### 3. **ANN Models (Python + Keras)**

Four specialized neural networks for different GA tasks:

#### **Model 1: Fitness Predictor**

```
Architecture:     50 → 128 → 64 → 32 → 1
Input Features:   48 (schedule features)
Output:           Float [0-100] (fitness score)
Purpose:          Rank schedules in GA population
Optimization:     MSE Loss, Adam optimizer
Expected R²:      > 0.85
Inference Time:   < 0.01ms per schedule
```

#### **Model 2: Constraint Classifier**

```
Architecture:     50 → 64 → 32 → 16 → 1
Input Features:   48 (schedule features)
Output:           Probability [0-1] (violation likelihood)
Purpose:          Detect risky/invalid schedules early
Optimization:     Binary Crossentropy, Adam
Expected Accuracy: > 90%
Use Case:         Filter invalid schedules during GA iterations
```

#### **Model 3: Crossover Recommender**

```
Architecture:     50 → 100 → 64 → 32 → 5
Input Features:   48 (schedule features)
Output:           5 crossover split point recommendations
Purpose:          Guide offspring creation
Optimization:     Categorical Crossentropy
Expected Top-5:   > 70% accuracy
Use Case:         Suggest optimal split points for breeding
```

#### **Model 4: Mutation Predictor**

```
Architecture:     50 → 64 → 32 → 1
Input Features:   48 (schedule features + mutation delta)
Output:           Probability [0-1] (mutation success)
Purpose:          Judge proposed mutations
Optimization:     Binary Crossentropy
Expected Accuracy: > 75%
Use Case:         Accept/reject mutations based on predictions
```

**Model Files Location**: [src/models.py](../src/models.py)  
**Training Pipelines**: [scripts/train\_\*.py](../scripts/)

### 4. **Python FastAPI Service**

```
FastAPI Application (src/api_service.py)
├─ Runs on: http://0.0.0.0:8000
├─ Loads at startup:
│  ├─ 4 Keras models (.keras files)
│  ├─ 4 Scalers (joblib .pkl files)
│  └─ Feature extractors
└─ Serves endpoints:
   ├─ POST /predict/fitness/batch
   ├─ POST /predict/constraints
   ├─ POST /recommend/crossover
   ├─ POST /predict/mutation
   ├─ GET /health
   └─ GET /models/status
```

**Dependencies**:

```
tensorflow (Keras)
fastapi
uvicorn
pydantic
numpy, pandas
joblib
```

### 5. **Go Backend (Genetic Algorithm)**

```
Go Application (C:\...\go-scheduling-backend\)
├─ Core GA Engine
│  ├─ Population initialization
│  ├─ Fitness ranking
│  ├─ Selection
│  ├─ Crossover
│  ├─ Mutation
│  └─ Elitism & termination
│
├─ ANN Integration Client
│  ├─ HTTP requests to Python API
│  ├─ Batch prediction capability
│  ├─ Error handling & retries
│  └─ Request/response marshaling
│
└─ Validation Engine
   ├─ Hard constraint checking
   ├─ Soft constraint scoring
   ├─ Schedule repair logic
   └─ Final schedule validation
```

**Key Files**:

- `SchedulePost.go` - Enables ANN mode
- `GeneticAlgorithm.go` - GA orchestration
- `ANNClient.go` - HTTP communication
- `Crossover.go` - Breeding with ANN guidance
- `ValidateIndividual.go` - Constraint validation

### 6. **Frontend/Client Layer**

```
Web Application (React/TypeScript)
├─ Schedule generation requests
├─ Parameter configuration
│  ├─ Population size
│  ├─ Mutation rate
│  ├─ Generations limit
│  └─ Enable/disable ANN
├─ Real-time progress display
├─ Results visualization
└─ Validation feedback
```

---

## Detailed Architecture

### System Block Diagram

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         HYBRID ANN-GA SCHEDULER                               │
│                                                                                │
│  ┌───────────────────┐                                                        │
│  │   FRONTEND        │  (Web Application)                                     │
│  │   ─────────────   │  - User interface                                      │
│  │   React/TS        │  - Configuration                                       │
│  └─────────┬─────────┘  - Results display                                     │
│            │                                                                  │
│      HTTP  │                                                                  │
│            ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐      │
│  │              GO BACKEND (Genetic Algorithm)                          │      │
│  │  ─────────────────────────────────────────────────────────────────  │      │
│  │                                                                      │      │
│  │  ┌──────────────────┐         ┌────────────────────────────────┐  │      │
│  │  │  GA Engine       │         │  Validation Engine            │  │      │
│  │  ├─ Population Init │         ├─ Hard Constraints Check       │  │      │
│  │  ├─ Selection       │         ├─ Soft Constraints Score       │  │      │
│  │  ├─ Crossover       │         ├─ Schedule Repair Logic        │  │      │
│  │  ├─ Mutation        │         └─ Final Validation             │  │      │
│  │  └─ Elitism         │                                          │  │      │
│  │                     │         ┌────────────────────────────────┐  │      │
│  │  ┌──────────────────┐         │  ANN Integration Client        │  │      │
│  │  │ Population       │         ├─ HTTP REST Client             │  │      │
│  │  │ Management       │         ├─ Batch Request Marshaling     │  │      │
│  │  │                  │         ├─ Response Parsing             │  │      │
│  │  │ [Individuals]    │         ├─ Error Handling & Retries     │  │      │
│  │  │ → Rank by        │         └─ Connection Management        │  │      │
│  │  │   Fitness        │                                          │  │      │
│  │  └──────────────────┘         Sends requests to Python API:  │  │      │
│  │            ↑                   • /predict/fitness/batch       │  │      │
│  │            │                   • /predict/constraints         │  │      │
│  │            │                   • /recommend/crossover         │  │      │
│  │            │                   • /predict/mutation            │  │      │
│  │            │                                                   │  │      │
│  │  ┌─────────┴──────────────────┐                               │  │      │
│  │  │ Validation + Repair        │                               │  │      │
│  │  │                            │                               │  │      │
│  │  │ If invalid or incomplete:  │                               │  │      │
│  │  │ • Add missing subjects     │                               │  │      │
│  │  │ • Fix time conflicts       │                               │  │      │
│  │  │ • Rebalance load           │                               │  │      │
│  │  └────────────────────────────┘                               │  │      │
│  │                                                                │  │      │
│  │  Output: Final optimized schedule                             │  │      │
│  └────────────────────┬─────────────────────────────────────────┘  │      │
│                       │                                             │      │
│              HTTP REST│ (predictions)                              │      │
│                       ▼                                             │      │
│  ┌──────────────────────────────────────────────────────────┐      │      │
│  │         PYTHON FASTAPI SERVICE (Port 8000)               │      │      │
│  │  ─────────────────────────────────────────────────────   │      │      │
│  │                                                          │      │      │
│  │  ┌──────────────────────────────────────────────────┐   │      │      │
│  │  │  Model Loading (Startup)                         │   │      │      │
│  │  ├─ Fitness Predictor (.keras)                      │   │      │      │
│  │  ├─ Constraint Classifier (.keras)                  │   │      │      │
│  │  ├─ Crossover Recommender (.keras)                  │   │      │      │
│  │  ├─ Mutation Predictor (.keras)                     │   │      │      │
│  │  ├─ 4 Scalers (StandardScaler .pkl)                 │   │      │      │
│  │  └─ Feature Extractors                              │   │      │      │
│  │                                                      │   │      │      │
│  │  ┌──────────────────────────────────────────────────┐   │      │      │
│  │  │  Request Handlers                                │   │      │      │
│  │  │                                                  │   │      │      │
│  │  │  1. /predict/fitness/batch                       │   │      │      │
│  │  │     ├─ Input: List[schedule_array]               │   │      │      │
│  │  │     ├─ Extract 48 features per schedule          │   │      │      │
│  │  │     ├─ Normalize with scaler                     │   │      │      │
│  │  │     ├─ Predict with fitness model                │   │      │      │
│  │  │     └─ Output: List[fitness_score]               │   │      │      │
│  │  │                                                  │   │      │      │
│  │  │  2. /predict/constraints                         │   │      │      │
│  │  │     ├─ Input: schedule_array                     │   │      │      │
│  │  │     ├─ Extract 48 features                       │   │      │      │
│  │  │     ├─ Normalize with scaler                     │   │      │      │
│  │  │     ├─ Predict violation probability             │   │      │      │
│  │  │     └─ Output: {violations, severity, details}   │   │      │      │
│  │  │                                                  │   │      │      │
│  │  │  3. /recommend/crossover                         │   │      │      │
│  │  │     ├─ Input: parent1, parent2 arrays            │   │      │      │
│  │  │     ├─ Extract features from both               │   │      │      │
│  │  │     ├─ Predict optimal split points              │   │      │      │
│  │  │     └─ Output: [split1, split2, ...]             │   │      │      │
│  │  │                                                  │   │      │      │
│  │  │  4. /predict/mutation                            │   │      │      │
│  │  │     ├─ Input: schedule, proposed_mutation        │   │      │      │
│  │  │     ├─ Extract features with delta               │   │      │      │
│  │  │     ├─ Predict mutation success probability      │   │      │      │
│  │  │     └─ Output: {prob, recommendation}            │   │      │      │
│  │  │                                                  │   │      │      │
│  │  └──────────────────────────────────────────────────┘   │      │      │
│  │                                                          │      │      │
│  │  ┌──────────────────────────────────────────────────┐   │      │      │
│  │  │  Health & Monitoring                             │   │      │      │
│  │  ├─ GET /health → {status, models_loaded}           │   │      │      │
│  │  ├─ GET /models/status → {model: load_time, ...}    │   │      │      │
│  │  └─ Logging & metrics collection                    │   │      │      │
│  │                                                      │   │      │      │
│  └──────────────────────────────────────────────────────┘   │      │      │
│                         ↑                                   │      │      │
│                         │ Models & Scalers                 │      │      │
│                         ▼                                   │      │      │
│  ┌──────────────────────────────────────────────────┐       │      │      │
│  │         TRAINING DATA PIPELINE                   │       │      │      │
│  │  ────────────────────────────────────────────   │       │      │      │
│  │                                                 │       │      │      │
│  │  Data Sources:                                 │       │      │      │
│  │  ├─ data/manual_schedules.json                 │       │      │      │
│  │  └─ data/manual_schedules_with_fitness.json    │       │      │      │
│  │                                                 │       │      │      │
│  │  Processing:                                   │       │      │      │
│  │  ├─ Generate synthetic variants                │       │      │      │
│  │  ├─ Augment with constraints/crossovers        │       │      │      │
│  │  ├─ Calculate fitness labels                   │       │      │      │
│  │  └─ Split train/val/test                       │       │      │      │
│  │                                                 │       │      │      │
│  │  Output:                                       │       │      │      │
│  │  └─ data/training_output/training_data.json    │       │      │      │
│  │                                                 │       │      │      │
│  │  Trained Models (saved to src/models/):        │       │      │      │
│  │  ├─ fitness_predictor.keras                    │       │      │      │
│  │  ├─ constraint_classifier.keras                │       │      │      │
│  │  ├─ crossover_recommender.keras                │       │      │      │
│  │  └─ mutation_predictor.keras                   │       │      │      │
│  │                                                 │       │      │      │
│  │  Scalers (saved to src/models/):               │       │      │      │
│  │  ├─ feature_scaler.pkl                         │       │      │      │
│  │  ├─ fitness_scaler.pkl                         │       │      │      │
│  │  ├─ constraint_scaler.pkl                      │       │      │      │
│  │  └─ mutation_scaler.pkl                        │       │      │      │
│  │                                                 │       │      │      │
│  └──────────────────────────────────────────────────┘       │      │      │
│                                                              │      │      │
└──────────────────────────────────────────────────────────────┘      │      │
                                                                      │      │
                         ┌─────────────────────┐                      │      │
                         │  EXTERNAL STORAGE   │                      │      │
                         ├─ File System        │                      │      │
                         │ src/models/         │                      │      │
                         │ data/training_output│                      │      │
                         │ logs/               │                      │      │
                         └─────────────────────┘                      │      │
                                                                      │      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### 1. **Training Data Generation Flow**

```
Manual Schedules (JSON)
        ↓
[import_existing_data.py]
        ↓
Raw schedule records
        ↓
[generate_synthetic_variants.py]
        ├─ Mutation variants
        ├─ Crossover variants
        └─ Augmented constraints
        ↓
Variant schedules
        ↓
[calculate_fitness_for_historical.py]
        ├─ Calculate fitness scores
        ├─ Label constraint violations
        └─ Mark mutation effectiveness
        ↓
Labeled training data (JSONL)
        ↓
[Train 4 Models]
        ├─ Split: 70% train, 15% val, 15% test
        ├─ Feature scaling (StandardScaler)
        ├─ Model training with callbacks
        └─ Save: models/*.keras + scalers/*.pkl
```

### 2. **Runtime Prediction Flow (GA Iteration)**

```
GA Population [Individual 1..N]
        ↓
[Go Backend]
        ├─ Batch fitness requests
        ├─ Mutation decisions
        ├─ Crossover guidance
        └─ Constraint checks
        ↓
HTTP POST to Python API
        ↓
[FastAPI Service]
        ├─ Extract features (48 dims)
        ├─ Normalize with scalers
        ├─ Predict with models
        └─ Return JSON response
        ↓
HTTP Response to Go
        ↓
[Go Backend]
        ├─ Rank population by fitness
        ├─ Select parents
        ├─ Apply crossover
        ├─ Apply mutations
        └─ Validate & repair
        ↓
Next Generation Population
```

### 3. **Model Inference Pipeline**

```
Individual Schedule (Go integer array)
        ↓
JSON Serialization
        ↓
HTTP POST /predict/fitness/batch
        ↓
[FastAPI]
        ├─ Parse JSON
        ├─ Create ScheduleFeatureExtractor
        ├─ extract_features(schedule)
        │  └─ Returns 48-dim numpy array
        │
        ├─ Load scaler
        ├─ scaler.transform(features)
        │  └─ Normalize each feature
        │
        ├─ Load fitness model
        ├─ model.predict(normalized_features)
        │  └─ Returns [0-100] fitness score
        │
        └─ Inverse transform if needed
        ↓
JSON Response with predictions
        ↓
Go backend processes response
```

---

## Integration Points

### 1. **Go ↔ Python Communication**

**Protocol**: HTTP/1.1 REST API  
**Format**: JSON request/response  
**Port**: 8000  
**Endpoints**:

| Endpoint                 | Method | Go Usage               | Latency Target |
| ------------------------ | ------ | ---------------------- | -------------- |
| `/predict/fitness/batch` | POST   | Rank population        | < 10ms (batch) |
| `/predict/constraints`   | POST   | Validate schedule      | < 5ms          |
| `/recommend/crossover`   | POST   | Guide breeding         | < 5ms          |
| `/predict/mutation`      | POST   | Accept/reject mutation | < 5ms          |
| `/health`                | GET    | Check service status   | < 1ms          |

**Go Client Code Location**:

```
Go Backend Repository
├─ GeneticAlgorithm/
│  └─ ANNClient.go
│     ├─ NewANNClient(baseURL)
│     ├─ PredictFitnessAsync(schedules)
│     ├─ PredictConstraints(schedule)
│     ├─ RecommendCrossover(parent1, parent2)
│     └─ PredictMutation(schedule, mutation)
```

### 2. **Configuration Integration**

**Python Config** ([src/config.py](../src/config.py)):

```python
# Model configurations
FITNESS_PREDICTOR_CONFIG = {
    'input_dim': 48,
    'hidden_dims': [128, 64, 32],
    'output_dim': 1,
    'loss': 'mse',
    'metrics': ['mae']
}

# Training settings
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 50,
    'validation_split': 0.2
}

# Feature settings
SCHEDULE_CONSTRAINTS = {
    'n_subjects': 5,
    'n_instructors': 3,
    'n_rooms': 3,
    'n_slots_per_day': 12
}
```

### 3. **Request/Response Marshaling**

**Go → Python (Fitness Batch)**:

```json
{
  "schedules": [
    [0, 0, 1, 1, 2, 2, ...],  // schedule 1 (50 integers)
    [0, 1, 1, 2, 2, 3, ...],  // schedule 2
    ...
  ]
}
```

**Python → Go (Fitness Response)**:

```json
{
  "success": true,
  "predictions": [85.3, 72.1, 90.5],
  "elapsed_ms": 8.5
}
```

---

## Deployment Structure

### Development Environment

```
Local Machine
├─ Terminal 1: Go Backend (runs GA)
│  └─ http://localhost:8080
│
├─ Terminal 2: Python API Service
│  └─ http://localhost:8000
│
├─ Terminal 3: Frontend Dev Server
│  └─ http://localhost:3000
│
└─ Workspace
   ├─ Go Backend (.go files)
   ├─ Python ANN (src/, scripts/)
   └─ Frontend (React/TS)
```

### Production Deployment

```
Server / Container Environment
├─ Kubernetes Pod / Docker Container
│  ├─ Go Backend Service
│  │  └─ Port 8080 (internal)
│  │
│  ├─ Python FastAPI Service
│  │  └─ Port 8000 (internal)
│  │
│  └─ Frontend (Static Assets)
│     └─ Port 3000 or 80 (external)
│
├─ Persistent Storage
│  ├─ Models volume
│  ├─ Data volume
│  └─ Logs volume
│
└─ Load Balancer / Ingress
   └─ Route external traffic
```

---

## Model Specifications

### Input/Output Specifications

#### Fitness Predictor

**Input:**

- Schedule array (integer): 50 elements representing schedule slots
- Features: 48 dimensions after extraction

**Output:**

- Float: fitness score [0.0 - 100.0]
- Interpretation: Higher = better schedule

**Training Data:**

- ~1,000-2,000 labeled schedules
- Balanced across fitness ranges

#### Constraint Classifier

**Input:**

- Schedule array (integer): 50 elements
- Features: 48 dimensions

**Output:**

- Float: probability [0.0 - 1.0]
- Interpretation: Likelihood of constraint violation

**Labels:**

- Valid schedule: 0
- Invalid schedule: 1

#### Crossover Recommender

**Input:**

- Parent1 array: 50 integers
- Parent2 array: 50 integers
- Features: 48 dimensions (combined)

**Output:**

- Array of 5 integers: split point recommendations
- Range: [0 - 50]

**Use:**

- Go backend tries each split point
- Selects one that produces highest fitness offspring

#### Mutation Predictor

**Input:**

- Original schedule: 50 integers
- Proposed mutation: delta information
- Features: 48 + delta features

**Output:**

- Float: probability [0.0 - 1.0]
- Interpretation: Likelihood mutation improves schedule

---

## API Endpoints

### 1. **Fitness Prediction (Batch)**

```
POST /predict/fitness/batch

Request:
{
  "schedules": [
    [0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 1, ...],
    [0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 1, 2, ...],
    ...
  ]
}

Response:
{
  "success": true,
  "predictions": [85.3, 72.1, 90.5, ...],
  "elapsed_ms": 12.5,
  "batch_size": 3
}

Error Response:
{
  "success": false,
  "error": "Invalid schedule length",
  "details": "Expected 50 integers, got 49"
}
```

### 2. **Constraint Prediction**

```
POST /predict/constraints

Request:
{
  "schedule": [0, 0, 1, 1, 2, 2, ...]
}

Response:
{
  "success": true,
  "violation_probability": 0.15,
  "has_violations": false,
  "violation_details": {
    "instructor_overload": false,
    "room_conflicts": false,
    "subject_imbalance": false,
    "early_classes": false
  }
}
```

### 3. **Crossover Recommendations**

```
POST /recommend/crossover

Request:
{
  "parent1": [0, 0, 1, 1, 2, 2, ...],
  "parent2": [0, 1, 1, 2, 2, 0, ...]
}

Response:
{
  "success": true,
  "recommended_splits": [10, 15, 20, 25, 30],
  "rationale": "Split points maximize genetic diversity",
  "expected_fitness": [78.2, 81.5, 79.3, 82.1, 80.4]
}
```

### 4. **Mutation Prediction**

```
POST /predict/mutation

Request:
{
  "schedule": [0, 0, 1, 1, 2, 2, ...],
  "mutation": {
    "index": 5,
    "new_value": 1,
    "old_value": 2
  }
}

Response:
{
  "success": true,
  "mutation_success_probability": 0.72,
  "recommendation": "ACCEPT",
  "fitness_delta_estimate": 2.3,
  "confidence": 0.85
}
```

### 5. **Health Check**

```
GET /health

Response:
{
  "status": "healthy",
  "models_loaded": 4,
  "scalers_loaded": 4,
  "fitness_predictor": "ready",
  "constraint_classifier": "ready",
  "crossover_recommender": "ready",
  "mutation_predictor": "ready",
  "startup_time_ms": 2345
}
```

### 6. **Model Status**

```
GET /models/status

Response:
{
  "models": {
    "fitness_predictor": {
      "loaded": true,
      "file": "src/models/fitness_predictor.keras",
      "load_time_ms": 523,
      "inference_time_ms_avg": 0.8
    },
    "constraint_classifier": {...},
    "crossover_recommender": {...},
    "mutation_predictor": {...}
  }
}
```

---

## Performance Characteristics

### Latency Targets

| Operation                     | Target | Typical | Notes                                    |
| ----------------------------- | ------ | ------- | ---------------------------------------- |
| Single fitness prediction     | < 1ms  | 0.8ms   | 48-dim feature extraction + NN inference |
| Batch fitness (100 schedules) | < 15ms | 12ms    | ~0.1ms overhead per item                 |
| Constraint check              | < 5ms  | 3.2ms   | Single schedule validation               |
| Crossover recommendation      | < 10ms | 7.5ms   | Dual-parent analysis                     |
| Mutation prediction           | < 5ms  | 4.2ms   | Single mutation analysis                 |
| API health check              | < 1ms  | 0.5ms   | No model inference                       |

### Resource Usage

**Python Service Memory**:

- Idle: ~500 MB
- With 4 models loaded: ~2.5 GB
- With 100-batch prediction: ~3 GB

**CPU**:

- Idle: 1-2%
- During prediction: 15-25% (single core)
- During batch (100): 40-60%

**Network**:

- Average request: 2-5 KB
- Average response: 1-3 KB
- Throughput: ~100-200 predictions/second (single connection)

### Scaling Considerations

1. **Horizontal Scaling (Multiple API Instances)**:
   - Load balance between multiple Python services
   - Each instance runs 4 models independently
   - Use Nginx/Kubernetes for distribution

2. **Batch Processing**:
   - Group requests from Go backend
   - Send 50-100 schedules per batch request
   - Reduces overhead from HTTP round-trips

3. **Model Optimization**:
   - Quantization (FP32 → INT8) for 4x speedup
   - TensorFlow Lite for edge deployment
   - ONNX format for broader runtime support

---

## System Monitoring & Logging

### Python API Logging

```
2026-05-16T14:32:15 [INFO] ann_api: STARTUP: loading models and scalers
2026-05-16T14:32:15 [INFO] ann_api: ✓ Loaded fitness predictor from src/models/fitness_predictor.keras
2026-05-16T14:32:16 [INFO] ann_api: ✓ Loaded constraint classifier from src/models/constraint_classifier.keras
...
2026-05-16T14:32:18 [INFO] uvicorn: Uvicorn running on http://0.0.0.0:8000
2026-05-16T14:32:45 [INFO] ann_api: POST /predict/fitness/batch - 50 schedules in 12.3ms
2026-05-16T14:32:45 [INFO] ann_api: Predictions: [85.2, 72.1, 90.5, ...]
```

### Go Backend Logging

```
[GA] Generation 1: Population size 50, Avg fitness 72.3
[ANN] Batch prediction (50 schedules) in 12.5ms
[GA] Crossover: Using recommended split at position 25
[GA] Mutation accepted (prob: 0.72, fitness delta: +2.3)
[GA] Generation complete: Best fitness 92.1
```

---

## Architecture Decision Records (ADRs)

### ADR-1: Python for ANN Service

**Decision**: Use Python + FastAPI for neural network predictions  
**Rationale**:

- TensorFlow/Keras ecosystem mature in Python
- Model training in Python significantly easier
- FastAPI provides fast HTTP interface
- Easy model versioning and updates

### ADR-2: Separate Process for ANN

**Decision**: Run ANN as separate service, communicate via HTTP  
**Rationale**:

- Decouples GA logic from model inference
- Allows independent scaling
- Easy deployment (Docker containers)
- Language flexibility (Python + Go)
- Can restart service without stopping GA

### ADR-3: Batch Predictions from Go

**Decision**: Go backend groups multiple predictions per request  
**Rationale**:

- Reduces HTTP overhead
- Enables TensorFlow batch optimization
- More efficient resource usage
- Fewer context switches

### ADR-4: Feature Normalization

**Decision**: Normalize features with StandardScaler at inference time  
**Rationale**:

- Consistent with training phase
- Ensures model stability
- Prevents numerical issues
- Allows for future feature scaling updates

---

## Troubleshooting & Common Issues

### Issue: API Service Won't Start

**Symptoms**: `Connection refused` or `Port 8000 already in use`

**Solutions**:

1. Check if port is in use: `netstat -ano | findstr :8000`
2. Kill existing process: `taskkill /PID <pid> /F`
3. Change port in config: `API_PORT = 8001`

### Issue: Model Predictions Seem Wrong

**Symptoms**: Fitness scores outside [0-100] range or constant values

**Solutions**:

1. Verify feature extraction: Check if schedule array is correct length (50)
2. Check scalers: Ensure scaler files are loaded
3. Retrain models: Models may be out of date with new schedule format
4. Check logs: Look for NaN or inf values in predictions

### Issue: Slow Predictions

**Symptoms**: Each prediction takes > 10ms

**Solutions**:

1. Use batch requests: Group predictions together
2. Check system load: CPU/memory usage
3. Monitor network latency: Between Go and Python
4. Consider model quantization: For faster inference

---

## Related Documentation

- [ANN_IMPLEMENTATION_GUIDE.md](ANN_IMPLEMENTATION_GUIDE.md) - Detailed model implementation
- [PROCESS_FLOW.md](PROCESS_FLOW.md) - Data preparation workflow
- [DATA_IMPORT_GUIDE.md](DATA_IMPORT_GUIDE.md) - How to import schedule data
- [GA_ANN_INTEGRATION_WORKING.md](../GA_ANN_INTEGRATION_WORKING.md) - Integration setup guide
- [README.md](../README.md) - Quick reference guide

---

**Last Updated**: May 16, 2026  
**Maintained By**: Development Team  
**Version**: 2.0.0
