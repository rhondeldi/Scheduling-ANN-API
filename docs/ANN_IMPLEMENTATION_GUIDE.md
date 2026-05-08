# ANN Model Implementation Guide for Scheduling System

## Overview

This guide outlines the process of building an Artificial Neural Network (ANN) to assist the Genetic Algorithm in generating optimal course schedules.

## Table of Contents

1. [System Analysis](#1-system-analysis)
2. [ANN Architecture Design](#2-ann-architecture-design)
3. [Data Preparation](#3-data-preparation)
4. [Model Development](#4-model-development)
5. [Integration Strategy](#5-integration-strategy)
6. [Training Process](#6-training-process)
7. [Evaluation & Optimization](#7-evaluation--optimization)

---

## 1. System Analysis

### Current GA System Components

- **TimeSlot**: Contains SubjectID, InstructorID, RoomID (each 16-bit)
- **Schedule Hierarchy**: UniTimeTables → WeekTimeTable → DayTimeTable → TimeSlot
- **Fitness Function**: Evaluates schedules based on:
  - Lunch break availability (12-3pm)
  - Classes after 5pm (penalized)
  - Hours per day (preferred ≤10 hours)
  - Days with classes (preferred ≤4 days)
  - Saturday long hours (penalized)

### ANN Purpose Options

We'll implement **1 ANN model** to assist the genetic algorithm:

- **Fitness Predictor ANN**: Fast fitness estimation (100x speedup)

---

## 2. ANN Architecture Design

### Model 1: Fitness Predictor Neural Network

**Purpose**: Predict schedule fitness without full evaluation (faster than fitness function)

**Input Features** (per schedule):

- Total weekly hours
- Distribution of hours per day (6-day vector)
- Number of days with classes
- Classes before 12pm count
- Classes during lunch (12-3pm) count
- Classes after 5pm count
- Saturday hours
- Total gaps between classes
- Instructor load distribution variance
- Room utilization rate
- Curriculum year level distribution

**Architecture**:

```
Input Layer:    50-100 neurons (encoded features)
Hidden Layer 1: 128 neurons (ReLU activation)
Hidden Layer 2: 64 neurons (ReLU activation)
Hidden Layer 3: 32 neurons (ReLU activation)
Output Layer:   1 neuron (Linear activation → fitness score)
```

**Loss Function**: Mean Squared Error (MSE)

---

## 3. Data Preparation

### Phase 1: Data Collection

**Training Data Sources**:

1. **Historical GA Runs**: Save all generated schedules with fitness scores
2. **Manual Expert Schedules**: Existing good schedules
3. **Synthetic Data**: Generate edge cases

**Data Collection Script**:

```python
# Collect during GA execution
generation_data = {
    'schedule': encoded_schedule,
    'fitness': fitness_score,
    'constraints_violated': constraint_list,
    'generation_number': gen,
    'parent_fitness': [parent1_fit, parent2_fit],
    'mutation_applied': mutation_type
}
```

### Phase 2: Feature Engineering

**Schedule Encoding Methods**:

1. **Flat Encoding**:
   - 6 days × 24 time_slots × 3 attributes (subject, instructor, room)
   - Total: 432 values per schedule

2. **Statistical Features**:
   - Daily hour totals (6 values)
   - Lunch break flags (6 values)
   - After-5pm flags (6 values)
   - Gap counts per day (6 values)
   - Instructor workload variance
   - Room utilization rates

3. **One-Hot Encoding**:
   - Subject types (lecture/lab)
   - Room types
   - Instructor preferences

### Phase 3: Normalization

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# For continuous features (hours, gaps)
scaler = StandardScaler()
normalized_features = scaler.fit_transform(continuous_features)

# For fitness scores (bound outputs)
minmax_scaler = MinMaxScaler(feature_range=(0, 1))
normalized_fitness = minmax_scaler.fit_transform(fitness_scores)
```

### Phase 4: Dataset Split

```python
train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15

# Stratified split based on fitness ranges (good/medium/poor schedules)
train_data, temp_data = train_test_split(data, test_size=0.3, stratify=fitness_ranges)
val_data, test_data = train_test_split(temp_data, test_size=0.5)
```

---

## 4. Model Development

### Technology Stack

**Framework**: TensorFlow/Keras or PyTorch

**Libraries**:

- NumPy: Numerical operations
- Pandas: Data handling
- Scikit-learn: Preprocessing & metrics
- Matplotlib/Seaborn: Visualization

### Implementation Steps

#### Step 1: Environment Setup

```bash
# Create virtual environment
python -m venv ann_env
source ann_env/bin/activate  # Windows: ann_env\Scripts\activate

# Install dependencies
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn
```

#### Step 2: Build Fitness Predictor Model

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

def build_fitness_predictor(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear')  # Fitness score
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae', 'mse']
    )

    return model
#### Step 3: Integration Setup

---

## 5. Integration Strategy

### Architecture Overview

```

┌─────────────────────────────────────────────────────────────┐
│ Scheduling System │
│ │
│ ┌──────────────┐ │
│ │ Go Backend │ │
│ │ (GA Engine) │ │
│ └───────┬───────┘ │
│ │ │
│ │ REST API / HTTP │
│ ▼ │
│ ┌──────────────────────────────────────┐ │
│ │ Python ANN Service │ │
│ │ (FastAPI/Flask Server) │ │
│ │ │ │
│ │ ┌──────────────────────────────┐ │ │
│ │ │ Model 1: Fitness Predictor │ │ │
│ │ └──────────────────────────────┘ │ │
│ │ │ │
│ └──────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

````

### Integration Methods

#### Option 1: REST API (Recommended)

**Python Service**:
```python
from fastapi import FastAPI
import numpy as np

app = FastAPI()

@app.post("/predict_fitness")
async def predict_fitness(schedule_data: dict):
    # Preprocess schedule
    features = extract_features(schedule_data)

    # Predict
    fitness = fitness_model.predict(features)

    return {"predicted_fitness": float(fitness[0])}
````

**Go Client**:

```go
func PredictFitness(schedule Schedule.UniTimeTables) (float64, error) {
    encoded := EncodeScheduleForANN(schedule)

    resp, err := http.Post(
        "http://localhost:8000/predict_fitness",
        "application/json",
        bytes.NewBuffer(encoded),
    )

    // Parse response
    var result map[string]float64
    json.NewDecoder(resp.Body).Decode(&result)

    return result["predicted_fitness"], nil
}
```

#### Option 2: gRPC (Recommended for production)

**Benefits**: Faster, type-safe, bidirectional streaming

#### Option 3: Embedded Python (Advanced)

Use cgo to call Python directly (more complex but lowest latency)

---

## 6. Training Process

### Phase 1: Initial Data Generation

```python
# pseudo_code.py
# Run GA multiple times to collect training data

num_training_runs = 100
training_data = []

for run in range(num_training_runs):
    # Run GA with different parameters
    ga_result = run_genetic_algorithm(
        population_size=random.randint(50, 200),
        generations=random.randint(100, 500),
        mutation_rate=random.uniform(0.01, 0.1)
    )

    # Collect all schedules generated
    for schedule, fitness in ga_result.all_individuals:
        training_data.append({
            'schedule': schedule,
            'fitness': fitness,
            'features': extract_features(schedule)
        })

# Save training data
save_training_data(training_data, 'training_dataset.pkl')
```

### Phase 2: Training Loop

```python
# train_fitness_predictor.py

def train_model(model, X_train, y_train, X_val, y_val):
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
        'models/fitness_predictor_best.h5',
        monitor='val_loss',
        save_best_only=True
    )

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )

    return history

# Execute training
model = build_fitness_predictor(input_dim=X_train.shape[1])
history = train_model(model, X_train, y_train, X_val, y_val)
```

### Phase 3: Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

def create_model(neurons=128, dropout=0.2, learning_rate=0.001):
    model = build_fitness_predictor_with_params(neurons, dropout)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# Grid search
param_grid = {
    'neurons': [64, 128, 256],
    'dropout': [0.1, 0.2, 0.3],
    'learning_rate': [0.0001, 0.001, 0.01],
    'batch_size': [16, 32, 64],
    'epochs': [100, 200]
}

keras_model = KerasRegressor(build_fn=create_model, verbose=0)
grid_search = GridSearchCV(estimator=keras_model, param_grid=param_grid, cv=3)
grid_result = grid_search.fit(X_train, y_train)

print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
```

---

## 7. Evaluation & Optimization

### Performance Metrics

#### For Fitness Predictor:

- **MSE** (Mean Squared Error): Lower is better
- **MAE** (Mean Absolute Error): Average prediction error
- **R² Score**: Goodness of fit (closer to 1 is better)
- **MAPE** (Mean Absolute Percentage Error): Percentage accuracy

```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
print(f"MAPE: {mape:.4f}%")
```

### Model Optimization Techniques

1. **Regularization**: L1/L2 to prevent overfitting
2. **Batch Normalization**: Stabilize training
3. **Dropout**: Random neuron deactivation
4. **Learning Rate Scheduling**: Adaptive learning
5. **Data Augmentation**: Generate synthetic schedules
6. **Ensemble Methods**: Combine multiple models

### Deployment Checklist

- [ ] Model achieves target R² > 0.85
- [ ] Inference time < 10ms per prediction
- [ ] Model size < 50MB (for deployment efficiency)
- [ ] API response time < 50ms (including network)
- [ ] Error handling implemented
- [ ] Logging and monitoring configured
- [ ] Model versioning system in place
- [ ] A/B testing framework ready
- [ ] Rollback strategy defined

---

## Next Steps

1. **Implement data collection pipeline** in GA backend
2. **Build and train fitness predictor** (highest priority)
3. **Create Python API service** for model serving
4. **Integrate with Go backend** via REST/gRPC
5. **Measure performance improvements** in GA
6. **Iterate and refine** based on results

---

## Expected Improvements

| Metric                  | Before ANN | After ANN  | Improvement             |
| ----------------------- | ---------- | ---------- | ----------------------- |
| Avg Fitness Evaluations | 50,000     | 10,000     | 80% reduction           |
| Time to Solution        | 60 seconds | 15 seconds | 75% faster              |
| Solution Quality        | Good       | Better     | 10-15% fitness increase |
| Convergence Speed       | 500 gens   | 100 gens   | 80% faster              |

---

## References & Further Reading

1. **Neural Networks for Combinatorial Optimization**: Bello et al. (2016)
2. **GA-NN Hybrid Systems**: Genetic Algorithm with Neural Network Guidance
3. **TensorFlow Documentation**: https://tensorflow.org/tutorials
4. **Keras API Reference**: https://keras.io/api/
5. **FastAPI Documentation**: https://fastapi.tiangolo.com/

---

**Version**: 1.0  
**Last Updated**: March 6, 2026  
**Author**: GitHub Copilot
