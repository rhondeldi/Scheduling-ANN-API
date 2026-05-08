# ANN Model Implementation Process Flow

This document shows the **step-by-step process** for building and deploying the ANN models.

## 📊 Complete Process Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ANN IMPLEMENTATION PROCESS                       │
└─────────────────────────────────────────────────────────────────────────┘

PHASE 1: SETUP & PREPARATION
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  [1] Environment Setup          [2] Data Collection                     │
│      │                               │                                   │
│      ├─ Install Python 3.8+          ├─ Collect from GA runs            │
│      ├─ Create venv                  ├─ Generate synthetic data         │
│      ├─ Install requirements.txt    └─ Save to data/*.json              │
│      └─ Create directories                                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
PHASE 2: DATA PREPARATION
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  [3] Feature Engineering        [4] Data Preprocessing                  │
│      │                               │                                   │
│      ├─ Extract schedule features    ├─ Normalize features               │
│      ├─ Statistical features         ├─ Train/Val/Test split            │
│      ├─ Constraint features          ├─ Save scalers                    │
│      └─ Temporal patterns            └─ Stratified sampling             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
PHASE 3: MODEL DEVELOPMENT
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  [5] Build Model                [6] Train Model                         │
│      │                               │                                   │
│      ├─ Fitness Predictor            ├─ Set callbacks                   │
│      │  └─ 50 → 128 → 64 → 32 → 1   ├─ Early stopping                  │
│      │                               ├─ Learning rate schedule          │
│      │                               ├─ Model checkpointing             │
│      │                               └─ TensorBoard logging             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
PHASE 4: EVALUATION & OPTIMIZATION
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  [7] Evaluate Performance       [8] Optimize                            │
│      │                               │                                   │
│      ├─ Test set metrics             ├─ Hyperparameter tuning           │
│      ├─ R² score, MAE, MSE           ├─ Architecture search             │
│      ├─ Confusion matrix             ├─ Feature selection               │
│      └─ Validation curves            └─ Ensemble methods                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
PHASE 5: DEPLOYMENT
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  [9] API Service                [10] Integration                        │
│      │                                │                                  │
│      ├─ Start FastAPI server          ├─ REST API client (Go)           │
│      ├─ Load trained models           ├─ gRPC (optional)                │
│      ├─ Serve predictions             ├─ Batch processing               │
│      └─ Health monitoring             └─ Error handling                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
PHASE 6: PRODUCTION USE
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  [11] GA Integration            [12] Monitoring                         │
│       │                               │                                  │
│       ├─ Fitness prediction           ├─ Track performance              │
│       └─ Handle predictions           ├─ Log predictions                │
│                                       └─ Model drift detection          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow Diagram

```
┌──────────────┐
│  GA Backend  │  (Go)
│   Runs GA    │
└──────┬───────┘
       │
       │ 1. Schedule + Fitness
       ▼
┌────────────────────┐
│  Data Collection   │
│   (data_collection.py)
│                    │
│  Saves:            │
│  • Schedule data   │
│  • Fitness scores  │
│  • Constraints     │
└──────┬─────────────┘
       │
       │ 2. Training Data
       ▼
┌────────────────────┐
│ Feature Extraction │
│ (feature_extraction.py)
│                    │
│  Extracts:         │
│  • 50 features     │
│  • Normalized      │
└──────┬─────────────┘
       │
       │ 3. Feature Vectors
       ▼
┌────────────────────┐
│   Model Training   │
│ (train_*.py)       │
│                    │
│  Trains:           │
│  • 200 epochs      │
│  • Validation      │
│  • Checkpoints     │
└──────┬─────────────┘
       │
       │ 4. Trained Models
       ▼
┌────────────────────┐
│   API Service      │
│ (api_service.py)   │
│                    │
│  Endpoint:         │
│  • /predict/fitness│
└──────┬─────────────┘
       │
       │ 5. Predictions (HTTP)
       ▼
┌────────────────────┐
│  GA Backend        │
│ (Go Integration)   │
│                    │
│  Uses:             │
│  • Quick fitness   │
│  • Guided search   │
└────────────────────┘
```

## 📝 Detailed Step-by-Step Procedure

### Step 1: Environment Setup (5 minutes)

```bash
# 1. Check Python version
python --version  # Should be 3.8+

# 2. Create project directory
cd scheduling-ANN-model

# 3. Create virtual environment
python -m venv venv

# 4. Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 5. Install dependencies
pip install -r requirements.txt
```

**Expected Output**: All packages installed successfully

---

### Step 2: Data Collection (Variable time)

**Option A: Collect Real Data** (Recommended for production)

```go
// In your Go backend, add data collection
func CollectTrainingData(schedule Schedule.UniTimeTables, fitness float64) {
    data := map[string]interface{}{
        "schedule": convertScheduleToArray(schedule),
        "fitness": fitness,
        "timestamp": time.Now(),
    }

    // Save to JSON file
    saveToJSON("training_data.json", data)
}
```

**Option B: Generate Synthetic Data** (For testing)

```bash
python data_collection.py
```

**Expected Output**:

- `data/training_data_*.json` created
- 1000+ samples collected

---

### Step 3: Feature Engineering (Already implemented)

The `feature_extraction.py` extracts **50 features**:

1. **Temporal (12)**: Daily hours, weekly hours, distribution
2. **Constraints (12)**: Lunch breaks, late classes per day
3. **Resources (8)**: Instructor/room utilization, load balance
4. **Distribution (10)**: Gaps, compactness, spread
5. **Workload (8)**: Morning/afternoon/evening ratios

**No action needed** - features are extracted automatically during training.

---

### Step 4: Model Training (15-30 minutes per model)

```bash
# Train fitness predictor
python train_fitness_predictor.py
```

**Monitor Progress**:

- Watch loss decrease
- Check validation metrics
- Training stops when validation loss plateaus

**Expected Results**:

- R² > 0.85
- MAE < 5.0
- Model saved to `models/fitness_predictor.h5`

---

### Step 5: Model Evaluation

```python
# Evaluation is automatic during training
# Check plots in logs/ directory
# Review metrics in console output
```

**Key Metrics**:

- **Loss**: Should converge to low value
- **MAE**: Mean Absolute Error
- **R²**: Goodness of fit (closer to 1 is better)

---

### Step 6: API Deployment (2 minutes)

```bash
# Start API service
python api_service.py

# Or use uvicorn for production
uvicorn api_service:app --host 0.0.0.0 --port 8000 --workers 4
```

**Test API**:

```bash
# Health check
curl http://localhost:8000/health

# Should return:
# {
#   "status": "healthy",
#   "models_loaded": {
#     "fitness_predictor": true,
#     ...
#   }
# }
```

---

### Step 7: Go Integration (10 minutes)

1. **Copy integration client**:

   ```bash
   cp go_integration_client.go ../scheduling-system-backend/
   ```

2. **Modify your GA code**:

   ```go
   // In GeneticAlgorithm.go
   annClient := NewANNClient("http://localhost:8000")

   // Use ANN for fitness prediction
   fitness, err := annClient.PredictFitness(scheduleArray)
   ```

3. **Test integration**:
   ```bash
   cd ../scheduling-system-backend
   go run main.go
   ```

---

### Step 8: Production Deployment (Optional)

**Use Docker** for easier deployment:

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "api_service:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t ann-service .
docker run -p 8000:8000 ann-service
```

---

## 🎯 Quick Start (Automated)

Run the automated setup script:

```bash
python setup.py
```

This will:

1. ✓ Check Python version
2. ✓ Create directories
3. ✓ Set up virtual environment
4. ✓ Install dependencies
5. ✓ Generate synthetic data
6. ✓ Display model architectures

---

## 📈 Expected Performance Improvements

| Metric                  | Before ANN      | After ANN       | Improvement       |
| ----------------------- | --------------- | --------------- | ----------------- |
| **Fitness Evaluations** | 50,000          | 10,000          | **80% reduction** |
| **Evaluation Speed**    | 1ms each        | 0.01ms each     | **100x faster**   |
| **Time to Solution**    | 60 seconds      | 15 seconds      | **75% faster**    |
| **Convergence**         | 500 generations | 100 generations | **80% faster**    |

---

## ⚠️ Common Issues & Solutions

### Issue: Model not loading

**Solution**: Train the model first

```bash
python train_fitness_predictor.py
```

### Issue: API not responding

**Solution**: Check if service is running

```bash
netstat -ano | findstr :8000  # Windows
lsof -i :8000  # Linux/Mac
```

### Issue: Low prediction accuracy

**Solution**: Collect more training data (aim for 5,000+ samples)

### Issue: Out of memory

**Solution**: Reduce batch size in `config.py`

```python
FITNESS_PREDICTOR_CONFIG['batch_size'] = 16
```

---

## 🔍 Verification Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed from requirements.txt
- [ ] Training data collected (1,000+ samples)
- [ ] Fitness predictor trained (R² > 0.85)
- [ ] API service running and healthy
- [ ] Health check successful (http://localhost:8000/health)
- [ ] Go integration client copied
- [ ] Test prediction successful
- [ ] Models saved in models/ directory
- [ ] Scalers saved (feature_scaler.joblib, fitness_scaler.joblib)

---

## 📚 Next Steps

1. **Collect Real Data**: Replace synthetic data with actual GA runs
2. **Train All Models**: Not just fitness predictor
3. **Fine-tune Hyperparameters**: Optimize for your specific use case
4. **Monitor Performance**: Track prediction accuracy over time
5. **Iterate**: Retrain with new data periodically

---

**For detailed technical information**, see [ANN_IMPLEMENTATION_GUIDE.md](ANN_IMPLEMENTATION_GUIDE.md)

**For API documentation**, visit http://localhost:8000/docs when service is running
