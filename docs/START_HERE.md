# Step-by-Step Process: From Zero to Trained ANN Model

## 📋 Overview

This guide walks you through the complete process of setting up and training your ANN model using manual historical schedules.

**Time estimate**: 30-60 minutes (depending on data size)

---

## ✅ Prerequisites

Before starting, ensure you have:

- [ ] Python 3.8+ installed
- [ ] Manual/historical schedule files (JSON format preferred)
- [ ] Access to the scheduling-ANN-model directory

---

## 🚀 STEP-BY-STEP PROCESS

### STEP 1: Setup Python Environment (5 minutes)

Open PowerShell and navigate to the project directory:

```powershell
# Navigate to the ANN model directory
cd c:\Users\Sam\Desktop\Scheduling-System-ANN-GA\scheduling-ANN-model

# Verify you're in the right place
ls
```

**Expected output**: You should see files like `setup.py`, `models.py`, `README.md`, etc.

---

### STEP 2: Install Dependencies (5-10 minutes)

Run the automated setup script:

```powershell
python setup.py
```

**What this does**:

- Creates virtual environment
- Installs TensorFlow, NumPy, Pandas, FastAPI
- Creates necessary directories
- Verifies installation

**Expected output**:

```
✓ Virtual environment created
✓ Dependencies installed
✓ Directories created
✓ Setup complete!
```

**If you get errors**, manually install:

```powershell
pip install tensorflow numpy pandas matplotlib scikit-learn fastapi uvicorn
```

---

### STEP 3: Prepare Your Manual Schedule Data (10-15 minutes)

#### Option A: Your schedules are already in JSON

If your schedules look like this:

```json
{
  "schedules": [
    {
      "id": "section_1",
      "week_schedule": [
        [ /* Monday - 24 slots */ ],
        [ /* Tuesday - 24 slots */ ],
        ...
      ]
    }
  ]
}
```

✅ You're ready! Note the file path.

#### Option B: Your schedules are in another format

You need to convert them to JSON. Common conversions:

**From Excel/CSV:**

```powershell
# Use example_import.py or write a custom converter
python example_import.py
```

**From database:**

- Export to JSON
- Use the format from `example_manual_schedule_format.json`

**Need help with your format?** Check `example_manual_schedule_format.json` for the expected structure.

---

### STEP 4: Calculate Fitness Scores (5 minutes)

Since your manual schedules don't have fitness scores, calculate them:

```powershell
# Replace 'path\to\your\file.json' with your actual file path
python calculate_fitness_for_historical.py path\to\your\manual_schedules.json
```

**Example with actual path:**

```powershell
# If your file is in the parent directory:
python calculate_fitness_for_historical.py ..\scheduling-system-backend\manual_schedules.json

# If it's in the current directory:
python calculate_fitness_for_historical.py manual_schedules.json

# If it's on your Desktop:
python calculate_fitness_for_historical.py C:\Users\Sam\Desktop\manual_schedules.json
```

**Expected output**:

```
📂 Loading schedules from: manual_schedules.json
✓ Loaded 50 schedule(s)

🧮 Calculating fitness scores...
   ✓ Schedule 0 (ID: section_1): Fitness = 12.50
   ✓ Schedule 1 (ID: section_2): Fitness = 18.30
   ...

📊 Statistics:
   • Total schedules: 50
   • Fitness range: 8.50 - 35.20
   • Average fitness: 16.75
   • Median fitness: 15.30

🎯 Quality Distribution:
   • Excellent (< 20): 35 (70.0%)
   • Good (20-40): 13 (26.0%)
   • Fair (40-60): 2 (4.0%)

💾 Saved 50 schedules with fitness scores to: manual_schedules_with_fitness.json
```

**This creates**: `manual_schedules_with_fitness.json` (your original file + calculated fitness scores)

---

### STEP 5: Import Into Training Data (2 minutes)

Import the schedules with fitness scores into the training format:

```powershell
python -c "from import_existing_data import ScheduleDataImporter; i = ScheduleDataImporter(); i.import_from_json('manual_schedules_with_fitness.json'); i.print_statistics(); i.save_training_data()"
```

**Alternative (if the above doesn't work)**: Create a file called `do_import.py`:

```python
from import_existing_data import ScheduleDataImporter

importer = ScheduleDataImporter()
importer.import_from_json('manual_schedules_with_fitness.json')
importer.print_statistics()
importer.save_training_data()
print("\n✅ Import complete!")
```

Then run:

```powershell
python do_import.py
```

**Expected output**:

```
Importing schedules from: manual_schedules_with_fitness.json
✓ Imported 50 schedules

Statistics:
  • Total samples: 50
  • Fitness range: 8.50 - 35.20
  • Average fitness: 16.75

✓ Saved to: data/training_data.json
```

**This creates**: `data/training_data.json` (ready for training)

---

### STEP 6: Verify Training Data (1 minute)

Check that the training data was created correctly:

```powershell
# Check if file exists
Test-Path data\training_data.json

# Count samples (should show number of schedules)
python -c "import json; data = json.load(open('data/training_data.json')); print(f'Training samples: {len(data)}')"
```

**Expected output**:

```
True
Training samples: 50
```

---

### STEP 7: Train the ANN Model (10-20 minutes)

Start training the fitness predictor model:

```powershell
python train_fitness_predictor.py
```

**What happens**:

1. Loads training data
2. Extracts features from schedules
3. Splits into training/validation sets
4. Trains neural network
5. Saves trained model
6. Generates plots

**Expected output**:

```
Loading training data...
✓ Loaded 50 samples

Extracting features...
✓ Extracted 53 features per schedule

Building model...
✓ Model architecture:
   Input: 53 features
   Hidden: 128 → 64 → 32 neurons
   Output: 1 (fitness score)

Training model...
Epoch 1/100
  loss: 125.34 - mae: 8.92 - val_loss: 98.45 - val_mae: 7.12
Epoch 2/100
  loss: 87.23 - mae: 6.45 - val_loss: 76.34 - val_mae: 5.89
...
Epoch 100/100
  loss: 12.45 - mae: 2.34 - val_loss: 15.67 - val_mae: 2.89

✓ Training complete!

Model saved to: models/fitness_predictor.keras

Evaluation:
  • Training MAE: 2.34
  • Validation MAE: 2.89
  • Training R²: 0.95
  • Validation R²: 0.92

✅ Model ready for use!
```

**Training time**:

- Small dataset (50-100): 5-10 minutes
- Medium dataset (100-500): 10-15 minutes
- Large dataset (500+): 15-30 minutes

**This creates**:

- `models/fitness_predictor.keras` (trained model)
- `results/training_history.png` (training curves)
- `results/predictions_vs_actual.png` (accuracy plot)

---

### STEP 8: Test the Model (2 minutes)

Verify the model works:

```powershell
python -c "from tensorflow import keras; import numpy as np; model = keras.models.load_model('models/fitness_predictor.keras'); print('✓ Model loaded successfully'); print(f'Model input shape: {model.input_shape}'); print(f'Model output shape: {model.output_shape}')"
```

**Expected output**:

```
✓ Model loaded successfully
Model input shape: (None, 53)
Model output shape: (None, 1)
```

---

### STEP 9: Start the API Service (Optional, 2 minutes)

Start the REST API to serve predictions:

```powershell
python api_service.py
```

**Expected output**:

```
Loading models...
✓ Loaded fitness predictor model

Starting FastAPI server...
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.

Available endpoints:
  • POST /predict/fitness
  • GET /health
  • GET /docs (API documentation)
```

**Test it in another PowerShell window:**

```powershell
# Test health endpoint
Invoke-RestMethod -Uri http://localhost:8000/health
```

**Expected response**:

```json
{
  "status": "healthy",
  "models_loaded": ["fitness_predictor"]
}
```

---

### STEP 10: View Results (2 minutes)

Check the generated plots:

```powershell
# Open results folder
explorer results\

# View training history
explorer results\training_history.png

# View predictions accuracy
explorer results\predictions_vs_actual.png
```

**What to look for**:

- **Training history**: Loss should decrease over epochs
- **Predictions plot**: Points should be close to the diagonal line
- **MAE < 5**: Model is accurate
- **R² > 0.85**: Model explains variance well

---

## ✅ SUCCESS CHECKLIST

After completing all steps, verify:

- [ ] ✅ Virtual environment created
- [ ] ✅ Dependencies installed
- [ ] ✅ Manual schedules found and formatted
- [ ] ✅ Fitness scores calculated
- [ ] ✅ Training data created (`data/training_data.json`)
- [ ] ✅ Model trained successfully
- [ ] ✅ Model file exists (`models/fitness_predictor.keras`)
- [ ] ✅ Training plots generated
- [ ] ✅ API service starts (optional)

---

## 🎯 What You've Accomplished

You now have:

1. ✅ **Trained ANN model** that predicts schedule fitness
2. ✅ **~100x faster** fitness evaluation vs full calculation
3. ✅ **REST API** ready to integrate with Go backend
4. ✅ **Training pipeline** ready to add more data later

---

## 🔄 Next Steps

### Immediate Next Steps:

1. **Integrate with Go Backend**

   ```go
   // Copy go_integration_client.go to your Go project
   // Use it to call the ANN API for fitness predictions
   ```

2. **Collect More Data**
   - Run your GA and save generated schedules
   - Add them to training data
   - Retrain for improved accuracy

3. **Deploy API**
   - Keep `api_service.py` running
   - Configure Go backend to use it
   - Test end-to-end integration

### Advanced Improvements:

1. **Add more training data**
   - Mix manual + GA schedules
   - Aim for 1000+ samples
   - Retrain periodically

2. **Train other models**
   - Constraint classifier
   - Crossover recommender
   - Mutation predictor

3. **Monitor performance**
   - Track prediction accuracy
   - Compare ANN vs actual fitness
   - Fine-tune as needed

---

## 🆘 Troubleshooting

### Issue: Setup fails

**Solution**:

```powershell
# Manually create environment
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install tensorflow numpy pandas matplotlib scikit-learn fastapi uvicorn
```

### Issue: Fitness calculation fails

**Solution**: Check your JSON format matches `example_manual_schedule_format.json`

### Issue: Training data import fails

**Solution**: Ensure fitness scores were calculated first

### Issue: Training takes too long

**Solution**: Reduce epochs in `train_fitness_predictor.py` (line ~200, change 100 to 50)

### Issue: Model accuracy is poor (MAE > 10)

**Solution**:

- Need more training data (aim for 100+ samples)
- Check if schedules are diverse enough
- Verify fitness scores are correct

### Issue: API won't start

**Solution**:

```powershell
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Use different port
python api_service.py --port 8001
```

---

## 📚 Documentation Reference

- **Full Implementation**: [ANN_IMPLEMENTATION_GUIDE.md](ANN_IMPLEMENTATION_GUIDE.md)
- **Data Import Details**: [DATA_IMPORT_GUIDE.md](DATA_IMPORT_GUIDE.md)
- **Manual Schedules Info**: [MANUAL_SCHEDULES_GUIDE.md](MANUAL_SCHEDULES_GUIDE.md)
- **Quick Reference**: [README.md](README.md)

---

## 📞 Quick Commands Summary

```powershell
# 1. Setup
cd c:\Users\Sam\Desktop\Scheduling-System-ANN-GA\scheduling-ANN-model
python setup.py

# 2. Calculate fitness
python calculate_fitness_for_historical.py your_schedules.json

# 3. Import training data
python -c "from import_existing_data import ScheduleDataImporter; i = ScheduleDataImporter(); i.import_from_json('your_schedules_with_fitness.json'); i.save_training_data()"

# 4. Train model
python train_fitness_predictor.py

# 5. Start API
python api_service.py
```

---

**🎉 Congratulations!** You've successfully set up and trained your ANN model for schedule fitness prediction!
