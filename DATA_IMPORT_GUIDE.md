# How to Import Your Existing Schedule Datasets

## 📊 Yes, You NEED the Datasets!

**The ANN models require training data to learn patterns.** Your existing schedule datasets are **perfect** for this - even better than synthetic data!

### 🌟 Manual Schedules from Instructors = EXCELLENT Training Data!

**If your historical schedules were manually created by instructors**, they are **IDEAL** for training:

- ✅ **Already optimized** - Human experts created them
- ✅ **Real constraints** - Reflects actual institutional needs
- ✅ **Proven quality** - These schedules worked in practice
- ✅ **High quality** - Usually have very low fitness scores (few violations)

**Don't worry if they don't have fitness scores** - we can calculate them for you! See the [Quick Start for Manual Schedules](#-quick-start-manual-schedules-without-fitness-scores) section below.

### Why Historical Data is Better

| Synthetic Data          | Historical Data (Your Datasets) |
| ----------------------- | ------------------------------- |
| Random patterns         | Real-world patterns             |
| May not reflect reality | Proven to work                  |
| Generic                 | Specific to your institution    |
| Quick to generate       | Already available ✓             |

## 🎯 What Data Do You Need?

### Minimum Requirements

1. **Schedules**: Week schedules (6 days × 24 time slots × 3 attributes)
2. **Fitness Scores**: The fitness/quality score for each schedule
3. **Quantity**: Ideally 1,000+ schedules (more is better)

### Data Format Required

```json
{
  "schedules": [
    {
      "id": "schedule_1",
      "fitness": 45.2,
      "semester": 1,
      "week_schedule": [
        [ // Monday
          [subject_id, instructor_id, room_id],  // 7:00-7:30am
          [subject_id, instructor_id, room_id],  // 7:30-8:00am
          [2, 5, 3],  // 8:00-8:30am - Subject 2, Instructor 5, Room 3
          ...
        ],
        [ // Tuesday
          ...
        ],
        ... // Wednesday through Saturday
      ]
    },
    {
      "id": "schedule_2",
      "fitness": 38.7,
      ...
    }
  ]
}
```

---

## 🚀 Quick Start: Manual Schedules WITHOUT Fitness Scores

**If your historical schedules are from instructors and don't have fitness scores**, follow this simple workflow:

### Step 1: Prepare Your Schedule File

Make sure your schedules are in JSON format (or convert them to JSON):

```json
{
  "schedules": [
    {
      "id": "sem1_section1",
      "week_schedule": [
        [
          /* Monday - 24 slots */
        ],
        [
          /* Tuesday - 24 slots */
        ],
        [
          /* Wednesday - 24 slots */
        ],
        [
          /* Thursday - 24 slots */
        ],
        [
          /* Friday - 24 slots */
        ],
        [
          /* Saturday - 24 slots */
        ]
      ]
    }
  ]
}
```

### Step 2: Calculate Fitness Scores

Run our fitness calculator on your schedules:

```powershell
# Windows PowerShell
cd c:\Users\Sam\Desktop\Scheduling-System-ANN-GA\scheduling-ANN-model

python calculate_fitness_for_historical.py path\to\your\manual_schedules.json
```

This will:

- ✓ Calculate fitness for each schedule
- ✓ Analyze constraint violations
- ✓ Show quality statistics
- ✓ Save as `manual_schedules_with_fitness.json`

**Expected result**: Manual schedules typically have **low fitness scores** (10-30) because they were carefully crafted by humans!

### Step 3: Import into Training Data

```powershell
python -c "from import_existing_data import ScheduleDataImporter; i = ScheduleDataImporter(); i.import_from_json('manual_schedules_with_fitness.json'); i.print_statistics(); i.save_training_data()"
```

### Step 4: Train the Model

```powershell
python train_fitness_predictor.py
```

**That's it!** Your manual schedules are now training the ANN to recognize quality patterns.

---

## 📥 How to Import Your Datasets

### Method 1: JSON Format (Easiest)

If your schedules are in JSON format:

```python
from import_existing_data import ScheduleDataImporter

# Create importer
importer = ScheduleDataImporter()

# Import from JSON file
importer.import_from_json("path/to/your/schedules.json")

# Check what was imported
importer.print_statistics()

# Save to training format
importer.save_training_data()
```

### Method 2: Binary .sched Files (From Go Backend)

If you have `.sched` files from your Go backend:

```python
from import_existing_data import ScheduleDataImporter

importer = ScheduleDataImporter()

# You need to provide fitness scores for each section
fitness_scores = {
    0: 45.2,  # Section 0 fitness
    1: 38.7,  # Section 1 fitness
    2: 42.1,  # Section 2 fitness
    # ... more sections
}

# Import binary file
importer.import_from_binary(
    binary_file_path="path/to/univ-sem-1.sched",
    num_sections=50,  # Total sections in file
    fitness_scores=fitness_scores
)

importer.save_training_data()
```

### Method 3: GA Execution Logs

If you have logs from past GA runs:

```python
from import_existing_data import ScheduleDataImporter

importer = ScheduleDataImporter()

# Import from log directory
# Automatically finds all JSON files with schedules and fitness
importer.import_from_ga_execution_logs("path/to/ga_logs/")

importer.print_statistics()
importer.save_training_data()
```

### Method 4: CSV Export

If you exported schedules to CSV:

```python
from import_existing_data import ScheduleDataImporter

importer = ScheduleDataImporter()

# Import from CSV
importer.import_from_csv("path/to/schedules.csv")

importer.save_training_data()
```

### Method 5: Multiple Sources Combined

You can combine data from multiple sources:

```python
from import_existing_data import ScheduleDataImporter

importer = ScheduleDataImporter()

# Import from multiple sources
importer.import_from_json("schedules_2024.json")
importer.import_from_json("schedules_2025.json")
importer.import_from_ga_execution_logs("ga_runs/")

# All data is combined
importer.print_statistics()
importer.save_training_data("data/all_schedules.json")
```

## 🔧 Step-by-Step: Import Your Data

### Step 1: Locate Your Schedule Files

Find where your past schedules are stored. Common locations:

```
scheduling-system-backend/
  └── scheduling-system-temporary-data/
      ├── univ-sem-1.sched
      ├── univ-sem-2.sched
      └── univ-sem-3.sched
```

### Step 2: Check Your Data Format

Run this to inspect your data:

```python
import json

# If JSON format
with open('your_schedule.json', 'r') as f:
    data = json.load(f)
    print(json.dumps(data, indent=2)[:500])  # Print first 500 chars
```

### Step 3: Create Import Script

Create a file `my_import.py`:

```python
from import_existing_data import ScheduleDataImporter

# Initialize importer
importer = ScheduleDataImporter()

# Import your data (adjust path)
importer.import_from_json("../scheduling-system-backend/scheduling-system-temporary-data/schedules.json")

# Or if binary:
# fitness_scores = {...}  # You need to calculate/provide these
# importer.import_from_binary("univ-sem-1.sched", num_sections=50, fitness_scores=fitness_scores)

# Check what was imported
importer.print_statistics()

# Save for training
importer.save_training_data()

print("\n✓ Data import complete!")
print("Now you can train the model: python train_fitness_predictor.py")
```

### Step 4: Run Import

```bash
cd c:\Users\Sam\Desktop\Scheduling-System-ANN-GA\scheduling-ANN-model
python my_import.py
```

### Step 5: Verify Import

Check that data was created:

```bash
# Check file exists
ls data/training_data.json

# If Windows:
dir data\training_data.json
```

## 📊 What About Fitness Scores?

### If You Have Fitness Scores

Great! Include them in your JSON:

```json
{
  "schedules": [
    {
      "id": "sched_1",
      "fitness": 45.2, // ← Your fitness score
      "week_schedule": [[[]]]
    }
  ]
}
```

### If You DON'T Have Fitness Scores ⭐ (MOST COMMON)

**Use our automatic calculator!** We provide a Python utility that calculates fitness using the same logic as your Go backend.

```powershell
# Calculate fitness for your historical schedules
python calculate_fitness_for_historical.py your_schedules.json

# This creates: your_schedules_with_fitness.json
```

**What it does:**

- ✅ Analyzes each schedule for constraint violations
- ✅ Calculates lunch break violations
- ✅ Detects late classes (after 5 PM)
- ✅ Measures daily workload distribution
- ✅ Counts gaps between classes
- ✅ Assigns fitness score (lower = better quality)

**Manual schedules typically score 10-30** (very good!) because they were carefully made by humans.

#### Alternative: Calculate in Python

If you want to customize the fitness calculation:

```python
from import_existing_data import ScheduleDataImporter
# Import your Go fitness function (you'll need to expose it via API or call it)

importer = ScheduleDataImporter()

# Import schedules without fitness
importer.import_from_json("schedules_no_fitness.json")

# Calculate fitness for each schedule
for sample in importer.training_samples:
    schedule = sample['schedule']['week_schedule']

    # Option 1: Call your Go fitness function via API
    fitness = call_go_fitness_api(schedule)

    # Option 2: Re-implement fitness in Python (or)
    fitness = calculate_fitness_python(schedule)

    sample['fitness'] = fitness

# Save with fitness scores
importer.save_training_data()
```

## 🎯 Common Scenarios

### ⭐ Scenario 1: Manual/Historical Schedules WITHOUT Fitness Scores (YOUR CASE!)

**You have**: JSON schedules created by instructors, no fitness scores

**Solution**: Use our automatic fitness calculator (3 commands!)

```powershell
# 1. Calculate fitness scores
python calculate_fitness_for_historical.py manual_schedules.json

# 2. Import the schedules with calculated fitness
python -c "from import_existing_data import ScheduleDataImporter; i = ScheduleDataImporter(); i.import_from_json('manual_schedules_with_fitness.json'); i.save_training_data()"

# 3. Train the model
python train_fitness_predictor.py
```

**Why this is great:**

- ✅ Manual schedules are high-quality (low fitness scores)
- ✅ They teach the ANN what "good" schedules look like
- ✅ Better than GA-generated schedules for training

---

### Scenario 2: I have JSON files with fitness scores

**Solution**: Use `import_from_json()` directly

```bash
python -c "
from import_existing_data import ScheduleDataImporter
importer = ScheduleDataImporter()
importer.import_from_json('my_schedules.json')
importer.save_training_data()
"
```

### Scenario 3: I have .sched binary files but no fitness

**Solution**: Calculate fitness first, then import

1. Load schedules in Go backend
2. Calculate fitness for each
3. Export to JSON with fitness
4. Import the JSON

Or create a Go script to export JSON:

```go
// export_schedules.go
func ExportSchedulesWithFitness(schedFile string, outputJSON string) {
    // Load schedules
    schedules := LoadSchedules(schedFile)

    // Calculate fitness for each
    result := []map[string]interface{}{}
    for i, sched := range schedules {
        fitness := MeasureUniSchedBasicFitness(sched, ...)

        result = append(result, map[string]interface{}{
            "id": fmt.Sprintf("section_%d", i),
            "fitness": fitness,
            "week_schedule": ConvertToArray(sched),
        })
    }

    // Save to JSON
    jsonData, _ := json.Marshal(result)
    ioutil.WriteFile(outputJSON, jsonData, 0644)
}
```

### Scenario 3: I have multiple semesters of data

**Solution**: Import all of them

```python
from import_existing_data import ScheduleDataImporter

importer = ScheduleDataImporter()

# Import each semester
for semester in [1, 2, 3]:
    importer.import_from_json(f"schedules_semester_{semester}.json")

# Combined dataset
print(f"Total: {len(importer.training_samples)} samples")
importer.save_training_data("data/all_semesters.json")
```

## ⚠️ Important Notes

### Data Quality Matters

- **Clean data**: Remove duplicate or invalid schedules
- **Diverse data**: Include good and bad schedules
- **Enough data**: Aim for 1,000+ samples minimum

### Fitness Score Requirements

- Must be consistent across all schedules
- Use the same fitness function
- Scores should reflect actual schedule quality

### Schedule Format Validation

The importer validates:

- ✓ 6 days (Monday-Saturday)
- ✓ 24 time slots per day
- ✓ 3 attributes per slot (subject, instructor, room)

Invalid schedules are automatically skipped.

## 🚀 After Import: Train the Model

Once data is imported:

```bash
# 1. Verify data exists
python -c "import json; print(len(json.load(open('data/training_data.json'))['schedules']) if 'schedules' in json.load(open('data/training_data.json')) else len(json.load(open('data/training_data.json'))))"

# 2. Train the model
python train_fitness_predictor.py

# 3. Start API
python api_service.py
```

## 📚 Example: Complete Workflow

```python
"""
complete_import_workflow.py
Complete example of importing existing data and training
"""

from import_existing_data import ScheduleDataImporter
import json

# Step 1: Initialize importer
print("Step 1: Initializing importer...")
importer = ScheduleDataImporter()

# Step 2: Import your data
print("Step 2: Importing schedules...")

# Adjust these paths to your actual data location
importer.import_from_json("../scheduling-system-backend/schedules_2024.json")
importer.import_from_json("../scheduling-system-backend/schedules_2025.json")

# Step 3: Check statistics
print("\nStep 3: Data statistics:")
importer.print_statistics()

# Step 4: Save for training
print("\nStep 4: Saving training data...")
importer.save_training_data()

# Step 5: Verify
print("\nStep 5: Verifying saved data...")
with open('data/training_data.json', 'r') as f:
    data = json.load(f)
    print(f"✓ Saved {len(data)} training samples")

print("\n" + "="*70)
print("SUCCESS! Data is ready for training.")
print("Next step: python train_fitness_predictor.py")
print("="*70)
```

Run it:

```bash
python complete_import_workflow.py
```

## 🆘 Troubleshooting

### Issue: "File not found"

**Solution**: Check file path. Use absolute path:

```python
import os
full_path = os.path.abspath("your_file.json")
importer.import_from_json(full_path)
```

### Issue: "Invalid schedule format"

**Solution**: Check your schedule structure:

```python
# Should be: 6 days × 24 slots × 3 attributes
print(len(schedule))  # Should be 6
print(len(schedule[0]))  # Should be 24
print(len(schedule[0][0]))  # Should be 3
```

### Issue: "No fitness scores"

**Solution**: You MUST provide fitness. Either:

1. Include in JSON
2. Calculate and provide as dict
3. Re-run GA to get fitness scores

### Issue: "Not enough data"

**Solution**: Need minimum 1,000 samples. Options:

1. Collect more historical data
2. Run GA multiple times to generate data
3. Use synthetic data temporarily (not ideal)

## 📞 Need Help?

If your data format is different, create an issue with:

1. Sample of your data format
2. Where it's located
3. What format you need it in

We can create a custom importer for your specific case.

---

**Summary**: Import your existing schedules → Provide fitness scores → Train models → Much better than synthetic data! 🎯
