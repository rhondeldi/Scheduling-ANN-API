"""
QUICK START - Execute these commands in order
Copy and paste each block into PowerShell
"""

print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║           ANN MODEL TRAINING - QUICK START COMMANDS                  ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

Follow these steps in order. Copy-paste each command into PowerShell.

""")

import sys
from pathlib import Path

def check_step(step_num, description, check_func, instruction):
    """Check if a step is completed"""
    print(f"\n{'='*70}")
    print(f"STEP {step_num}: {description}")
    print(f"{'='*70}")
    
    if check_func():
        print(f"✅ Already completed")
        return True
    else:
        print(f"❌ Not completed yet")
        print(f"\n📝 Instructions:")
        print(instruction)
        return False

# Check current directory
current_dir = Path.cwd()
print(f"Current directory: {current_dir}")

if "scheduling-ANN-model" not in str(current_dir):
    print("\n⚠️  You're not in the correct directory!")
    print("\nRun this command first:")
    print("cd c:\\Users\\Sam\\Desktop\\Scheduling-System-ANN-GA\\scheduling-ANN-model")
    sys.exit(1)

print("\n✅ You're in the correct directory\n")

# Step 1: Setup
step1_complete = check_step(
    1,
    "Install Dependencies",
    lambda: Path("venv").exists() or Path(".venv").exists(),
    """
Run this command:
    python setup.py

If that fails, run:
    pip install tensorflow numpy pandas matplotlib scikit-learn fastapi uvicorn
"""
)

# Step 2: Find your manual schedules
print(f"\n{'='*70}")
print(f"STEP 2: Locate Your Manual Schedule File")
print(f"{'='*70}")
print("""
Where is your manual schedule JSON file?

Common locations:
  • In parent directory: ..\\manual_schedules.json
  • In backend folder: ..\\scheduling-system-backend\\schedules.json
  • On Desktop: C:\\Users\\Sam\\Desktop\\schedules.json

📝 Note the file path - you'll need it for the next step
""")

# Step 3: Calculate fitness
step3_complete = check_step(
    3,
    "Calculate Fitness Scores",
    lambda: any(Path(".").glob("*_with_fitness.json")),
    """
Run this command (replace YOUR_FILE.json with your actual file):
    python calculate_fitness_for_historical.py YOUR_FILE.json

Examples:
    python calculate_fitness_for_historical.py ..\\manual_schedules.json
    python calculate_fitness_for_historical.py C:\\Users\\Sam\\Desktop\\schedules.json

This creates: YOUR_FILE_with_fitness.json
"""
)

# Step 4: Import training data
step4_complete = check_step(
    4,
    "Import Training Data",
    lambda: Path("data/training_data.json").exists(),
    """
Run this command (use the _with_fitness.json file from Step 3):

Simple version (create temp file first):
    
    echo "from import_existing_data import ScheduleDataImporter
i = ScheduleDataImporter()
i.import_from_json('YOUR_FILE_with_fitness.json')
i.print_statistics()
i.save_training_data()
print('\\n✅ Import complete!')" > do_import.py

    python do_import.py

Remember to replace YOUR_FILE_with_fitness.json with your actual filename!
"""
)

# Step 5: Train model
step5_complete = check_step(
    5,
    "Train the Model",
    lambda: Path("models/fitness_predictor.keras").exists() or Path("models/fitness_predictor.h5").exists(),
    """
Run this command:
    python train_fitness_predictor.py

This takes 10-20 minutes depending on your data size.
"""
)

# Step 6: Start API
print(f"\n{'='*70}")
print(f"STEP 6: Start API Service (Optional)")
print(f"{'='*70}")
print("""
To start the API service for Go backend integration:
    python api_service.py

This will run in the foreground. Open a new PowerShell window for other commands.
""")

# Summary
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")

completed = sum([step1_complete, step3_complete, step4_complete, step5_complete])
total = 4

print(f"\n📊 Progress: {completed}/{total} steps completed")

if completed == total:
    print("\n🎉 ALL STEPS COMPLETE!")
    print("\n✅ Your ANN model is trained and ready to use!")
    print("\n🚀 Next steps:")
    print("   1. Start API: python api_service.py")
    print("   2. Integrate with Go backend")
    print("   3. Test predictions")
else:
    print(f"\n📝 You have {total - completed} step(s) remaining")
    print("\nFollow the instructions above for each incomplete step.")

print(f"\n{'='*70}")
print("For detailed information, see: START_HERE.md")
print(f"{'='*70}\n")
