"""
Data Import Utility for Converting Existing Schedules to ANN Training Format

This script helps you import your existing schedule datasets into the format
required for training the ANN models.
"""

import json
import numpy as np
import struct
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import src.config as config
from src.feature_extraction import FitnessFeatureExtractor
from tqdm import tqdm


class ScheduleDataImporter:
    """
    Import existing schedule data from various formats
    """
    
    def __init__(self):
        self.feature_extractor = FitnessFeatureExtractor()
        self.training_samples = []
        
    def import_from_json(self, json_file_path: str, fitness_scores: Optional[Dict[str, float]] = None):
        """
        Import schedules from JSON file
        
        Args:
            json_file_path: Path to JSON file containing schedules
            fitness_scores: Optional dict mapping schedule IDs to fitness scores
                           If None, you need to provide fitness separately
        
        JSON Format Expected:
        {
            "schedules": [
                {
                    "id": "schedule_1",
                    "fitness": 45.2,  # Optional if fitness_scores provided
                    "semester": 1,
                    "week_schedule": [
                        [ # Day 0 (Monday)
                            [subject_id, instructor_id, room_id],  # Time slot 0
                            [subject_id, instructor_id, room_id],  # Time slot 1
                            ...
                        ],
                        [ # Day 1 (Tuesday)
                            ...
                        ],
                        ...
                    ]
                },
                ...
            ]
        }
        """
        print(f"Importing from JSON: {json_file_path}")
        
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        schedules = data.get('schedules', [])
        
        for schedule in tqdm(schedules, desc="Processing schedules"):
            schedule_id = schedule.get('id', 'unknown')
            
            # Get fitness score
            if 'fitness' in schedule:
                fitness = schedule['fitness']
            elif fitness_scores and schedule_id in fitness_scores:
                fitness = fitness_scores[schedule_id]
            else:
                print(f"Warning: No fitness score for schedule {schedule_id}, skipping...")
                continue
            
            week_schedule = schedule.get('week_schedule', [])
            
            if not self._validate_schedule(week_schedule):
                print(f"Warning: Invalid schedule format for {schedule_id}, skipping...")
                continue
            
            # Create training sample
            sample = {
                'schedule': {
                    'week_schedule': week_schedule
                },
                'fitness': fitness,
                'schedule_id': schedule_id,
                'metadata': {
                    'semester': schedule.get('semester'),
                    'source': 'json_import'
                }
            }
            
            self.training_samples.append(sample)
        
        print(f"Successfully imported {len(self.training_samples)} schedules")
        return len(self.training_samples)
    
    def import_from_binary(self, binary_file_path: str, num_sections: int, 
                          fitness_scores: Dict[int, float]):
        """
        Import schedules from Go backend binary format (.sched files)
        
        Args:
            binary_file_path: Path to .sched file
            num_sections: Number of sections in the schedule
            fitness_scores: Dict mapping section index to fitness score
        
        Binary Format (from Go backend):
        - Each time slot: 3 x uint16 (6 bytes)
        - Order: subject_id, instructor_id, room_id
        - Layout: [section][day][time_slot]
        """
        print(f"Importing from binary: {binary_file_path}")
        
        with open(binary_file_path, 'rb') as f:
            binary_data = f.read()
        
        # Constants from Go backend
        N_WEEKLY_SCHOOL_DAYS = 6
        N_DAILY_TIME_SLOTS = 24
        TIME_SLOT_BYTE_SIZE = 6
        
        bytes_per_section = N_WEEKLY_SCHOOL_DAYS * N_DAILY_TIME_SLOTS * TIME_SLOT_BYTE_SIZE
        
        for section_idx in range(num_sections):
            if section_idx not in fitness_scores:
                print(f"Warning: No fitness score for section {section_idx}, skipping...")
                continue
            
            # Extract binary data for this section
            section_start = section_idx * bytes_per_section
            section_end = section_start + bytes_per_section
            section_data = binary_data[section_start:section_end]
            
            # Parse week schedule
            week_schedule = []
            
            for day in range(N_WEEKLY_SCHOOL_DAYS):
                day_schedule = []
                
                for time_slot in range(N_DAILY_TIME_SLOTS):
                    idx_2D_to_1D = (day * N_DAILY_TIME_SLOTS) + time_slot
                    byte_offset = idx_2D_to_1D * TIME_SLOT_BYTE_SIZE
                    
                    # Read 3 uint16 values (little-endian)
                    subject_id = struct.unpack('<H', section_data[byte_offset:byte_offset+2])[0]
                    instructor_id = struct.unpack('<H', section_data[byte_offset+2:byte_offset+4])[0]
                    room_id = struct.unpack('<H', section_data[byte_offset+4:byte_offset+6])[0]
                    
                    day_schedule.append([int(subject_id), int(instructor_id), int(room_id)])
                
                week_schedule.append(day_schedule)
            
            # Create training sample
            sample = {
                'schedule': {
                    'week_schedule': week_schedule
                },
                'fitness': fitness_scores[section_idx],
                'schedule_id': f"section_{section_idx}",
                'metadata': {
                    'section_index': section_idx,
                    'source': 'binary_import'
                }
            }
            
            self.training_samples.append(sample)
        
        print(f"Successfully imported {len(self.training_samples)} schedules from binary")
        return len(self.training_samples)
    
    def import_from_multiple_files(self, file_paths: List[str], 
                                   fitness_file: Optional[str] = None):
        """
        Import from multiple schedule files
        
        Args:
            file_paths: List of paths to schedule files
            fitness_file: Optional JSON file with fitness scores
                         Format: {"schedule_1": 45.2, "schedule_2": 38.7, ...}
        """
        fitness_scores = {}
        
        if fitness_file:
            with open(fitness_file, 'r') as f:
                fitness_scores = json.load(f)
        
        for file_path in tqdm(file_paths, desc="Processing files"):
            if file_path.endswith('.json'):
                self.import_from_json(file_path, fitness_scores)
            elif file_path.endswith('.sched'):
                # For binary files, you need to specify metadata
                print(f"Binary file {file_path} requires additional metadata")
                print("Use import_from_binary() directly with proper parameters")
        
        return len(self.training_samples)
    
    def import_from_ga_execution_logs(self, log_directory: str):
        """
        Import schedules from GA execution logs
        
        Expected directory structure:
        log_directory/
            ├── run_1/
            │   ├── generation_0.json
            │   ├── generation_1.json
            │   └── ...
            ├── run_2/
            │   └── ...
        
        Generation file format:
        {
            "generation": 0,
            "population": [
                {
                    "individual_id": 0,
                    "fitness": 42.5,
                    "schedule": [[[]]]
                }
            ]
        }
        """
        print(f"Importing from GA logs: {log_directory}")
        
        log_dir = Path(log_directory)
        
        if not log_dir.exists():
            print(f"Error: Directory {log_directory} does not exist")
            return 0
        
        # Find all JSON files
        json_files = list(log_dir.rglob("*.json"))
        
        for json_file in tqdm(json_files, desc="Processing log files"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                population = data.get('population', [])
                generation = data.get('generation', 0)
                
                for individual in population:
                    fitness = individual.get('fitness')
                    schedule = individual.get('schedule')
                    
                    if fitness is None or schedule is None:
                        continue
                    
                    if not self._validate_schedule(schedule):
                        continue
                    
                    sample = {
                        'schedule': {
                            'week_schedule': schedule
                        },
                        'fitness': fitness,
                        'schedule_id': f"{json_file.stem}_ind_{individual.get('individual_id', 0)}",
                        'metadata': {
                            'generation': generation,
                            'source': 'ga_log',
                            'log_file': str(json_file)
                        }
                    }
                    
                    self.training_samples.append(sample)
            
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        print(f"Successfully imported {len(self.training_samples)} schedules from logs")
        return len(self.training_samples)
    
    def import_from_csv(self, csv_file_path: str):
        """
        Import from CSV format (if you exported schedules to CSV)
        
        CSV Format:
        schedule_id,fitness,semester,schedule_json
        sched_1,45.2,1,"[[[]]]"
        sched_2,38.7,1,"[[[]]]"
        """
        import csv
        
        print(f"Importing from CSV: {csv_file_path}")
        
        with open(csv_file_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in tqdm(reader, desc="Processing CSV rows"):
                try:
                    schedule_id = row['schedule_id']
                    fitness = float(row['fitness'])
                    schedule_json = json.loads(row['schedule_json'])
                    
                    if not self._validate_schedule(schedule_json):
                        continue
                    
                    sample = {
                        'schedule': {
                            'week_schedule': schedule_json
                        },
                        'fitness': fitness,
                        'schedule_id': schedule_id,
                        'metadata': {
                            'semester': int(row.get('semester', 1)),
                            'source': 'csv_import'
                        }
                    }
                    
                    self.training_samples.append(sample)
                
                except Exception as e:
                    print(f"Error processing row {row.get('schedule_id', 'unknown')}: {e}")
        
        print(f"Successfully imported {len(self.training_samples)} schedules from CSV")
        return len(self.training_samples)
    
    def _validate_schedule(self, week_schedule: List) -> bool:
        """
        Validate schedule structure
        
        Expected: 6 days x 24 time slots x 3 attributes
        """
        if not isinstance(week_schedule, list):
            return False
        
        # Should have 6 days
        if len(week_schedule) != config.N_WEEKLY_SCHOOL_DAYS:
            return False
        
        for day_schedule in week_schedule:
            if not isinstance(day_schedule, list):
                return False
            
            # Should have 24 time slots
            if len(day_schedule) != config.N_DAILY_TIME_SLOTS:
                return False
            
            for time_slot in day_schedule:
                if not isinstance(time_slot, list):
                    return False
                
                # Should have 3 attributes (subject, instructor, room)
                if len(time_slot) != 3:
                    return False
        
        return True
    
    def save_training_data(self, output_file: str = None):
        """
        Save imported data to training file format
        """
        if not self.training_samples:
            print("No training samples to save!")
            return
        
        if output_file is None:
            output_file = config.DATA_DIR / "training_data.json"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.training_samples, f, indent=2)
        
        print(f"\nSaved {len(self.training_samples)} training samples to: {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about imported data
        """
        if not self.training_samples:
            return {"error": "No data imported"}
        
        fitnesses = [s['fitness'] for s in self.training_samples]
        
        stats = {
            'total_samples': len(self.training_samples),
            'fitness_stats': {
                'min': min(fitnesses),
                'max': max(fitnesses),
                'mean': np.mean(fitnesses),
                'median': np.median(fitnesses),
                'std': np.std(fitnesses)
            },
            'sources': {}
        }
        
        # Count by source
        for sample in self.training_samples:
            source = sample.get('metadata', {}).get('source', 'unknown')
            stats['sources'][source] = stats['sources'].get(source, 0) + 1
        
        return stats
    
    def print_statistics(self):
        """
        Print detailed statistics
        """
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("IMPORTED DATA STATISTICS")
        print("="*70)
        
        print(f"\nTotal Samples: {stats['total_samples']}")
        
        print("\nFitness Statistics:")
        for key, value in stats['fitness_stats'].items():
            print(f"  {key.capitalize():8s}: {value:.2f}")
        
        print("\nSources:")
        for source, count in stats['sources'].items():
            print(f"  {source:15s}: {count} samples ({count/stats['total_samples']*100:.1f}%)")
        
        print("\n" + "="*70)


def convert_go_schedule_to_array(go_schedule_file: str, output_file: str = None):
    """
    Helper function to convert a single Go schedule file to JSON array format
    
    Usage:
        convert_go_schedule_to_array(
            "path/to/univ-sem-1.sched",
            "path/to/output.json"
        )
    """
    importer = ScheduleDataImporter()
    
    # You need to provide fitness scores
    # This is a example - adjust based on your actual fitness evaluation
    print("Note: You need to provide fitness scores for each section")
    print("Calculating fitness using GA fitness function...")
    
    # TODO: Call your GA fitness function here
    # fitness_scores = { section_idx: calculate_fitness(...) }
    
    print("Please provide fitness scores or use import_with_fitness_calculation()")


# Example usage functions

def example_import_json():
    """Example: Import from JSON files"""
    importer = ScheduleDataImporter()
    
    # Import from single JSON file
    importer.import_from_json("path/to/schedules.json")
    
    # Get statistics
    importer.print_statistics()
    
    # Save to training format
    importer.save_training_data()


def example_import_ga_logs():
    """Example: Import from GA execution logs"""
    importer = ScheduleDataImporter()
    
    # Import from log directory
    importer.import_from_ga_execution_logs("path/to/ga_logs/")
    
    # Statistics
    importer.print_statistics()
    
    # Save
    importer.save_training_data()


def example_import_multiple_sources():
    """Example: Import from multiple sources"""
    importer = ScheduleDataImporter()
    
    # Import from JSON
    importer.import_from_json("schedules_2024.json")
    
    # Import from GA logs
    importer.import_from_ga_execution_logs("ga_runs/")
    
    # Import from CSV
    importer.import_from_csv("exported_schedules.csv")
    
    # Statistics
    importer.print_statistics()
    
    # Save combined dataset
    importer.save_training_data("data/combined_training_data.json")


if __name__ == "__main__":
    print("="*70)
    print("SCHEDULE DATA IMPORT UTILITY")
    print("="*70)
    print("\nThis utility helps you import existing schedule datasets")
    print("for training the ANN models.")
    print("\nSupported formats:")
    print("  1. JSON files")
    print("  2. Binary .sched files (from Go backend)")
    print("  3. GA execution logs")
    print("  4. CSV files")
    print("\nSee examples in this file for usage patterns.")
    print("="*70)
    
    # Interactive mode
    print("\n\nWould you like to run an import? (y/n)")
    # Uncomment and modify for your specific use case:
    # importer = ScheduleDataImporter()
    # importer.import_from_json("your_file.json")
    # importer.print_statistics()
    # importer.save_training_data()
