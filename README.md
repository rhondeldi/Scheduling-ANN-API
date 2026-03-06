# Scheduling System - ANN Model

Artificial Neural Network models to assist the Genetic Algorithm in generating optimal course schedules.

## 📋 Overview

This module provides four specialized ANN models:

1. **Fitness Predictor**: Fast fitness score estimation
2. **Constraint Classifier**: Predict constraint violations
3. **Crossover Recommender**: Suggest optimal crossover points
4. **Mutation Predictor**: Predict mutation impact

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Training Data

```bash
# Generate synthetic data for testing
python data_collection.py

# Or integrate with your Go backend to collect real data
# See integration_example.go
```

### 3. Train Models

```bash
# Train fitness predictor
python train_fitness_predictor.py

# Train other models (similar structure)
# python train_constraint_classifier.py
# python train_crossover_recommender.py
# python train_mutation_predictor.py
```

### 4. Start API Service

```bash
# Start the FastAPI server
python api_service.py

# Or use uvicorn directly
uvicorn api_service:app --host 0.0.0.0 --port 8000 --workers 4
```

### 5. Test API

```bash
# Health check
curl http://localhost:8000/health

# Test fitness prediction
curl -X POST http://localhost:8000/predict/fitness \
  -H "Content-Type: application/json" \
  -d @test_schedule.json
```

## 📁 Project Structure

```
scheduling-ANN-model/
│
├── config.py                      # Configuration settings
├── requirements.txt               # Python dependencies
│
├── feature_extraction.py          # Feature engineering
├── models.py                      # Model definitions
│
├── data_collection.py             # Collect training data
├── train_fitness_predictor.py    # Train fitness model
│
├── api_service.py                 # FastAPI service
├── go_integration_client.go       # Go client example
│
├── data/                          # Training datasets
│   └── training_data.json
│
├── models/                        # Trained models
│   ├── fitness_predictor.h5
│   ├── constraint_classifier.h5
│   ├── crossover_recommender.h5
│   ├── mutation_predictor.h5
│   ├── feature_scaler.joblib
│   └── fitness_scaler.joblib
│
└── logs/                          # Training logs
    └── fitness_predictor/
```

## 🔧 Configuration

Edit `config.py` to customize:

- Model hyperparameters
- Data paths
- API settings
- Schedule constants

```python
# Example customization
FITNESS_PREDICTOR_CONFIG = {
    'input_dim': 50,
    'hidden_layers': [128, 64, 32],
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 200,
}
```

## 🔌 Integration with Go Backend

### Option 1: REST API (Recommended for prototyping)

```go
package main

import (
    "bytes"
    "encoding/json"
    "net/http"
)

type ScheduleData struct {
    WeekSchedule [][][]int `json:"week_schedule"`
}

type FitnessPredictionResponse struct {
    PredictedFitness float64 `json:"predicted_fitness"`
    ProcessingTimeMs float64 `json:"processing_time_ms"`
}

func PredictFitness(schedule Schedule.UniTimeTables) (float64, error) {
    // Convert schedule to JSON format
    scheduleData := ScheduleData{
        WeekSchedule: ConvertScheduleToArray(schedule),
    }

    requestBody, _ := json.Marshal(map[string]interface{}{
        "schedule": scheduleData,
    })

    // Make API call
    resp, err := http.Post(
        "http://localhost:8000/predict/fitness",
        "application/json",
        bytes.NewBuffer(requestBody),
    )
    if err != nil {
        return 0, err
    }
    defer resp.Body.Close()

    // Parse response
    var result FitnessPredictionResponse
    json.NewDecoder(resp.Body).Decode(&result)

    return result.PredictedFitness, nil
}
```

### Option 2: Batch Processing

For better performance, batch multiple predictions:

```go
func PredictFitnessBatch(schedules []Schedule.UniTimeTables) ([]float64, error) {
    // Batch multiple schedules in one request
    // Reduces network overhead
}
```

### Option 3: Async Predictions

Use Go channels for non-blocking predictions:

```go
func PredictFitnessAsync(schedule Schedule.UniTimeTables, resultChan chan<- float64) {
    go func() {
        fitness, _ := PredictFitness(schedule)
        resultChan <- fitness
    }()
}
```

## 📊 Model Performance

### Expected Metrics (after training)

| Model                 | Metric         | Target | Current |
| --------------------- | -------------- | ------ | ------- |
| Fitness Predictor     | R² Score       | >0.85  | -       |
| Fitness Predictor     | MAE            | <5.0   | -       |
| Constraint Classifier | Accuracy       | >90%   | -       |
| Constraint Classifier | F1 Score       | >0.85  | -       |
| Crossover Recommender | Top-5 Accuracy | >70%   | -       |
| Mutation Predictor    | Accuracy       | >75%   | -       |

### Training Requirements

- **Dataset Size**: Minimum 1,000 samples per model
- **Training Time**: 5-30 minutes per model (GPU recommended)
- **Inference Time**: <10ms per prediction
- **Model Size**: <50MB each

## 🎯 Usage Examples

### 1. Predict Fitness

```python
from api_service import predict_fitness
import requests

schedule_data = {
    "schedule": {
        "week_schedule": [...],  # 6x24x3 array
    }
}

response = requests.post(
    "http://localhost:8000/predict/fitness",
    json=schedule_data
)

fitness = response.json()["predicted_fitness"]
print(f"Predicted fitness: {fitness}")
```

### 2. Check Constraints

```python
response = requests.post(
    "http://localhost:8000/predict/constraints",
    json=schedule_data
)

violations = response.json()["violations"]
for constraint, violated in violations.items():
    if violated:
        print(f"⚠️ Violation: {constraint}")
```

### 3. Get Crossover Recommendations

```python
crossover_request = {
    "parent1": {"week_schedule": [...]},
    "parent2": {"week_schedule": [...]},
    "parent1_fitness": 45.2,
    "parent2_fitness": 48.7
}

response = requests.post(
    "http://localhost:8000/recommend/crossover",
    json=crossover_request
)

recommended_points = response.json()["recommended_points"]
print(f"Recommended crossover points: {recommended_points}")
```

## 🔬 Advanced Features

### Hyperparameter Tuning

Use Optuna for automatic hyperparameter optimization:

```python
import optuna

def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    # ... train model and return validation loss
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

### Model Ensemble

Combine multiple models for better accuracy:

```python
predictions = []
for model in models:
    pred = model.predict(features)
    predictions.append(pred)

ensemble_prediction = np.mean(predictions, axis=0)
```

### Transfer Learning

Fine-tune pre-trained models on new data:

```python
base_model = keras.models.load_model('fitness_predictor.h5')

# Freeze early layers
for layer in base_model.layers[:-3]:
    layer.trainable = False

# Fine-tune on new data
base_model.fit(new_X, new_y, epochs=50)
```

## 📈 Performance Optimization

### 1. Use GPU Acceleration

```python
# Check GPU availability
import tensorflow as tf
print("GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Enable mixed precision for faster training
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

### 2. Optimize Inference

```python
# Convert to TensorFlow Lite for faster inference
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Use ONNX for cross-platform optimization
import tf2onnx
onnx_model = tf2onnx.convert.from_keras(model)
```

### 3. Batch Predictions

```python
# Predict multiple schedules at once
predictions = model.predict(batch_of_features)  # Much faster than loop
```

## 🐛 Troubleshooting

### Issue: Models not loading

**Solution**: Ensure models are trained first:

```bash
python train_fitness_predictor.py
```

### Issue: Out of memory during training

**Solution**: Reduce batch size in `config.py`:

```python
FITNESS_PREDICTOR_CONFIG['batch_size'] = 16
```

### Issue: API not responding

**Solution**: Check if service is running:

```bash
curl http://localhost:8000/health
```

### Issue: Low prediction accuracy

**Solution**:

1. Collect more training data
2. Tune hyperparameters
3. Engineer better features
4. Try different model architectures

## 📚 Documentation

- [Implementation Guide](ANN_IMPLEMENTATION_GUIDE.md) - Detailed implementation process
- [API Documentation](http://localhost:8000/docs) - Interactive API docs (when server running)
- [Model Architecture](models.py) - Model definitions and explanations

## 🤝 Contributing

To add new models or features:

1. Define model in `models.py`
2. Create training script
3. Add API endpoint in `api_service.py`
4. Update this README

## 📄 License

This is part of the Scheduling System project. See main repository for license information.

## 🔗 Related

- [Backend (Go)](../scheduling-system-backend/)
- [Frontend (React)](../scheduling-system-frontend/)
- [Main Documentation](../README.md)

## 📞 Support

For issues or questions:

- Open an issue in the main repository
- Check [ANN_IMPLEMENTATION_GUIDE.md](ANN_IMPLEMENTATION_GUIDE.md) for detailed procedures

---

**Last Updated**: March 6, 2026  
**Version**: 1.0.0  
**Author**: GitHub Copilot
