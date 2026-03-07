"""
WORKFLOW: Import Manual Schedules (Without Fitness Scores)

This is a step-by-step workflow for importing historical schedules
created by instructors that don't have fitness scores.

Your case: Manual schedules from instructors → Calculate fitness → Train ANN
"""

import json
import sys
from pathlib import Path


def main():
    print("="*70)
    print("MANUAL SCHEDULE IMPORT WORKFLOW")
    print("="*70)
    
    print("\n📋 This workflow will guide you through:")
    print("   1. Locating your manual schedule files")
    print("   2. Calculating fitness scores")
    print("   3. Importing into training data")
    print("   4. Training the ANN model")
    
    print("\n" + "="*70)
    print("STEP 1: Locate Your Manual Schedule Files")
    print("="*70)
    
    print("\nWhere are your manual schedules stored?")
    print("Common locations:")
    print("  • ../scheduling-system-backend/schedules/")
    print("  • ../manual_schedules/")
    print("  • Documents/schedules/")
    
    schedule_file = input("\nEnter path to your schedule JSON file: ").strip()
    schedule_file = schedule_file.replace('"', '').replace("'", "")  # Remove quotes if pasted
    
    if not Path(schedule_file).exists():
        print(f"\n❌ File not found: {schedule_file}")
        print("\nPlease ensure:")
        print("  1. The path is correct")
        print("  2. The file exists")
        print("  3. You're in the correct directory")
        sys.exit(1)
    
    print(f"\n✓ Found file: {schedule_file}")
    
    # Check file format
    print("\n" + "="*70)
    print("STEP 2: Validate File Format")
    print("="*70)
    
    try:
        with open(schedule_file, 'r') as f:
            data = json.load(f)
        
        # Check structure
        if isinstance(data, list):
            schedules = data
        elif 'schedules' in data:
            schedules = data['schedules']
        elif 'schedule' in data:
            schedules = [data]
        else:
            print(f"\n⚠️  Unknown JSON structure")
            print("Expected: {'schedules': [...]}")
            print(f"Got keys: {list(data.keys())}")
            
            fix = input("\nContinue anyway? (y/n): ").strip().lower()
            if fix != 'y':
                sys.exit(1)
            schedules = []
        
        print(f"\n✓ Format validated")
        print(f"   • Found {len(schedules)} schedule(s)")
        
        # Check if fitness scores exist
        has_fitness = False
        if schedules:
            has_fitness = 'fitness' in schedules[0]
        
        if has_fitness:
            print(f"   • Fitness scores: ✓ Already present")
            skip_calc = input("\n   Skip fitness calculation? (y/n): ").strip().lower()
            if skip_calc == 'y':
                print("\n   Skipping to Step 4...")
                import_and_train(schedule_file)
                return
        else:
            print(f"   • Fitness scores: ✗ Need to calculate")
        
    except json.JSONDecodeError as e:
        print(f"\n❌ Invalid JSON file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error reading file: {e}")
        sys.exit(1)
    
    # Calculate fitness
    print("\n" + "="*70)
    print("STEP 3: Calculate Fitness Scores")
    print("="*70)
    
    print("\nCalculating fitness scores for your manual schedules...")
    print("This uses the same fitness function as your Go backend.")
    
    try:
        from scripts.calculate_fitness_for_historical import calculate_and_save
        
        # Determine output file
        input_path = Path(schedule_file)
        output_file = str(input_path.parent / f"{input_path.stem}_with_fitness.json")
        
        print(f"\nInput:  {schedule_file}")
        print(f"Output: {output_file}")
        
        # Calculate
        results = calculate_and_save(schedule_file, output_file, verbose=True)
        
        print(f"\n✓ Fitness calculation complete!")
        print(f"   • {len(results)} schedules processed")
        
        # Update schedule_file to the one with fitness
        schedule_file = output_file
        
    except ImportError:
        print("\n❌ Error: calculate_fitness_for_historical.py not found")
        print("   Please ensure you're in the scheduling-ANN-model directory")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error calculating fitness: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Import and train
    import_and_train(schedule_file)


def import_and_train(schedule_file):
    """Import the schedules and start training"""
    
    print("\n" + "="*70)
    print("STEP 4: Import into Training Data")
    print("="*70)
    
    try:
        from scripts.import_existing_data import ScheduleDataImporter
        
        print(f"\nImporting from: {schedule_file}")
        
        importer = ScheduleDataImporter()
        importer.import_from_json(schedule_file)
        
        print("\n✓ Import successful!")
        importer.print_statistics()
        
        # Save training data
        importer.save_training_data()
        
        print("\n✓ Training data saved to: data/training_data.json")
        
    except ImportError:
        print("\n❌ Error: import_existing_data.py not found")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error importing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Train model
    print("\n" + "="*70)
    print("STEP 5: Train the ANN Model")
    print("="*70)
    
    print("\nReady to train the fitness predictor model!")
    print("\nOptions:")
    print("  1. Train now (recommended)")
    print("  2. Train later manually")
    
    choice = input("\nChoice (1 or 2): ").strip()
    
    if choice == "1":
        print("\n🚀 Starting training...")
        print("This may take a few minutes depending on your data size.\n")
        
        try:
            import os
            os.system("python train_fitness_predictor.py")
        except Exception as e:
            print(f"\n⚠️  Could not auto-start training: {e}")
            print("\nPlease run manually:")
            print("  python train_fitness_predictor.py")
    else:
        print("\n📝 To train later, run:")
        print("  python train_fitness_predictor.py")
    
    # Final summary
    print("\n" + "="*70)
    print("✅ WORKFLOW COMPLETE!")
    print("="*70)
    
    print("\n📊 Summary:")
    print("  ✓ Manual schedules loaded")
    print("  ✓ Fitness scores calculated")
    print("  ✓ Training data prepared")
    print("  ✓ Ready for model training")
    
    print("\n🎯 Next Steps:")
    print("  1. Train the model (if not done already)")
    print("  2. Start the API service: python api_service.py")
    print("  3. Integrate with Go backend")
    
    print("\n📚 Documentation:")
    print("  • DATA_IMPORT_GUIDE.md - Detailed import guide")
    print("  • ANN_IMPLEMENTATION_GUIDE.md - Full implementation details")
    print("  • README.md - Quick reference")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MANUAL SCHEDULE IMPORT WORKFLOW")
    print("For schedules created by instructors (without fitness scores)")
    print("="*70)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Workflow cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
