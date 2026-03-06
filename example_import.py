"""
EXAMPLE: Import Your Existing Schedule Datasets

This is a template script you can modify to import YOUR specific data.
Uncomment and modify the section that matches your data format.
"""

from import_existing_data import ScheduleDataImporter
import json
from pathlib import Path


def example_1_simple_json():
    """
    If you have schedules in JSON format with fitness scores
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Import from JSON with fitness scores")
    print("="*70)
    
    importer = ScheduleDataImporter()
    
    # TODO: Replace with your actual file path
    json_file = "path/to/your/schedules.json"
    
    # Import
    importer.import_from_json(json_file)
    
    # Show statistics
    importer.print_statistics()
    
    # Save to training format
    importer.save_training_data()
    
    print("\n✓ Complete! Now run: python train_fitness_predictor.py")


def example_2_binary_sched_files():
    """
    If you have .sched files from Go backend
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Import from binary .sched files")
    print("="*70)
    
    importer = ScheduleDataImporter()
    
    # TODO: You need to provide fitness scores
    # Option 1: If you have them in a file
    try:
        with open("fitness_scores.json", 'r') as f:
            fitness_scores = json.load(f)
    except:
        # Option 2: Calculate them or provide manually
        fitness_scores = {
            0: 45.2,
            1: 38.7,
            2: 42.1,
            # ... add more section fitness scores
        }
    
    # TODO: Replace with your actual .sched file
    binary_file = "../scheduling-system-backend/scheduling-system-temporary-data/univ-sem-1.sched"
    num_sections = 50  # TODO: Adjust to your actual number of sections
    
    # Import
    importer.import_from_binary(binary_file, num_sections, fitness_scores)
    
    # Statistics and save
    importer.print_statistics()
    importer.save_training_data()
    
    print("\n✓ Complete! Now run: python train_fitness_predictor.py")


def example_3_multiple_semesters():
    """
    If you have multiple semester data files
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Import from multiple semester files")
    print("="*70)
    
    importer = ScheduleDataImporter()
    
    # TODO: Adjust paths to your actual schedule files
    schedule_files = [
        "../scheduling-system-backend/schedules_sem1_2024.json",
        "../scheduling-system-backend/schedules_sem2_2024.json",
        "../scheduling-system-backend/schedules_sem1_2025.json",
    ]
    
    # Import each file
    for schedule_file in schedule_files:
        print(f"\nImporting: {schedule_file}")
        try:
            importer.import_from_json(schedule_file)
            print(f"✓ Successfully imported {schedule_file}")
        except Exception as e:
            print(f"✗ Failed to import {schedule_file}: {e}")
    
    # Combined statistics
    importer.print_statistics()
    
    # Save combined dataset
    importer.save_training_data("data/combined_all_semesters.json")
    
    print("\n✓ Complete! Now run: python train_fitness_predictor.py")


def example_4_ga_execution_logs():
    """
    If you saved GA execution logs with schedules and fitness
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Import from GA execution logs")
    print("="*70)
    
    importer = ScheduleDataImporter()
    
    # TODO: Replace with your log directory
    log_directory = "../ga_execution_logs"
    
    # Import all logs
    importer.import_from_ga_execution_logs(log_directory)
    
    # Statistics
    importer.print_statistics()
    
    # Save
    importer.save_training_data()
    
    print("\n✓ Complete! Now run: python train_fitness_predictor.py")


def example_5_custom_format():
    """
    If your data is in a custom format, here's how to convert it
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Import from custom format")
    print("="*70)
    
    importer = ScheduleDataImporter()
    
    # TODO: Load your data in whatever format it's in
    # Example: Load from database, API, custom file format, etc.
    
    # Let's say you have data in a custom format:
    custom_data = load_your_custom_format()  # Implement this
    
    # Convert to training samples
    for item in custom_data:
        # Extract schedule (must be 6 days x 24 slots x 3 attributes)
        week_schedule = convert_to_week_schedule(item)  # Implement this
        
        # Extract fitness
        fitness = item['fitness_score']  # Adjust field name
        
        # Create sample
        sample = {
            'schedule': {
                'week_schedule': week_schedule
            },
            'fitness': fitness,
            'schedule_id': item['id'],
            'metadata': {
                'source': 'custom_format',
                'year': item.get('year', 2024)
            }
        }
        
        importer.training_samples.append(sample)
    
    # Statistics and save
    importer.print_statistics()
    importer.save_training_data()
    
    print("\n✓ Complete! Now run: python train_fitness_predictor.py")


def load_your_custom_format():
    """TODO: Implement based on your data format"""
    # Example implementation
    return []


def convert_to_week_schedule(item):
    """TODO: Convert your data structure to 6x24x3 format"""
    # Example: Create empty schedule
    week_schedule = []
    for day in range(6):
        day_schedule = []
        for slot in range(24):
            day_schedule.append([0, 0, 0])  # [subject, instructor, room]
        week_schedule.append(day_schedule)
    return week_schedule


def interactive_import():
    """
    Interactive mode - asks what format you have
    """
    print("\n" + "="*70)
    print("INTERACTIVE DATA IMPORT")
    print("="*70)
    print("\nWhat format is your data in?")
    print("1. JSON files with schedules and fitness scores")
    print("2. Binary .sched files (from Go backend)")
    print("3. Multiple files (different semesters/years)")
    print("4. GA execution logs")
    print("5. Custom format (needs code modification)")
    print("0. Exit")
    
    choice = input("\nEnter choice (0-5): ").strip()
    
    if choice == "1":
        file_path = input("Enter JSON file path: ").strip()
        importer = ScheduleDataImporter()
        importer.import_from_json(file_path)
        importer.print_statistics()
        save = input("Save to training_data.json? (y/n): ").strip().lower()
        if save == 'y':
            importer.save_training_data()
            print("✓ Saved!")
    
    elif choice == "2":
        print("\nFor binary files, you need to provide:")
        print("  1. Path to .sched file")
        print("  2. Number of sections")
        print("  3. Fitness scores for each section")
        print("\nPlease modify example_2_binary_sched_files() with your values")
        print("and run: python example_import.py")
    
    elif choice == "3":
        print("\nModify example_3_multiple_semesters() with your file paths")
        print("and run: python example_import.py")
    
    elif choice == "4":
        log_dir = input("Enter GA logs directory path: ").strip()
        importer = ScheduleDataImporter()
        importer.import_from_ga_execution_logs(log_dir)
        importer.print_statistics()
        save = input("Save to training_data.json? (y/n): ").strip().lower()
        if save == 'y':
            importer.save_training_data()
            print("✓ Saved!")
    
    elif choice == "5":
        print("\nFor custom formats:")
        print("1. Modify example_5_custom_format() function")
        print("2. Implement load_your_custom_format()")
        print("3. Implement convert_to_week_schedule()")
        print("4. Run: python example_import.py")
    
    elif choice == "0":
        print("Goodbye!")
    
    else:
        print("Invalid choice!")


def quick_test():
    """
    Quick test to see if import is working
    Creates a small test dataset
    """
    print("\n" + "="*70)
    print("QUICK TEST: Creating test dataset")
    print("="*70)
    
    importer = ScheduleDataImporter()
    
    # Create a few test samples
    for i in range(5):
        import numpy as np
        
        week_schedule = []
        for day in range(6):
            day_schedule = []
            for slot in range(24):
                # Random schedule
                subject = np.random.randint(0, 10)
                instructor = np.random.randint(0, 20) if subject > 0 else 0
                room = np.random.randint(0, 15) if subject > 0 else 0
                day_schedule.append([int(subject), int(instructor), int(room)])
            week_schedule.append(day_schedule)
        
        sample = {
            'schedule': {
                'week_schedule': week_schedule
            },
            'fitness': float(np.random.normal(40, 10)),
            'schedule_id': f"test_{i}",
            'metadata': {
                'source': 'test_data'
            }
        }
        
        importer.training_samples.append(sample)
    
    print(f"Created {len(importer.training_samples)} test samples")
    importer.print_statistics()
    
    # Save
    importer.save_training_data("data/test_training_data.json")
    
    print("\n✓ Test complete!")
    print("Test data saved to: data/test_training_data.json")
    print("\nYou can now test training with:")
    print("  python train_fitness_predictor.py")


if __name__ == "__main__":
    print("="*70)
    print("SCHEDULE DATA IMPORT - EXAMPLES")
    print("="*70)
    print("\nThis script shows examples of importing your schedule data.")
    print("Uncomment and run the example that matches your data format.")
    print("\nAvailable examples:")
    print("  1. example_1_simple_json()")
    print("  2. example_2_binary_sched_files()")
    print("  3. example_3_multiple_semesters()")
    print("  4. example_4_ga_execution_logs()")
    print("  5. example_5_custom_format()")
    print("  6. interactive_import() - Interactive mode")
    print("  7. quick_test() - Create test data")
    
    print("\n" + "="*70)
    print("What would you like to do?")
    print("="*70)
    print("1. Run interactive import")
    print("2. Run quick test (create test data)")
    print("3. Show me the code (exit and read this file)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        interactive_import()
    elif choice == "2":
        quick_test()
    else:
        print("\nTo import your data:")
        print("1. Open this file: example_import.py")
        print("2. Find the example matching your data format")
        print("3. Modify paths and parameters")
        print("4. Uncomment the example function call")
        print("5. Run: python example_import.py")
        print("\nOr read DATA_IMPORT_GUIDE.md for detailed instructions")
