"""
Calculate Fitness Scores for Historical Schedules

This utility calculates fitness scores for historical/manual schedules that don't have them.
Implements the same fitness function as your Go backend.

USAGE:
    python calculate_fitness_for_historical.py path/to/schedules.json output.json
"""

import json
import numpy as np
from pathlib import Path
import sys
from typing import List, Dict, Any, Tuple


class FitnessCalculator:
    """
    Calculates fitness scores for schedules using the same logic as Go backend.
    Based on: MeasureUniSchedBasicFitness function
    """
    
    def __init__(self):
        # Time slot constants (matching Go backend)
        self.SLOTS_PER_DAY = 24
        self.DAYS_PER_WEEK = 6
        
        # Lunch break slots (11:00 AM - 1:00 PM)
        # Slot 0 = 7:00 AM, so slot 8 = 11:00 AM, slot 12 = 1:00 PM
        self.LUNCH_START_SLOT = 8  # 11:00 AM
        self.LUNCH_END_SLOT = 12   # 1:00 PM
        
        # Late class threshold (after 5:00 PM)
        # Slot 20 = 5:00 PM
        self.LATE_CLASS_SLOT = 20  # 5:00 PM
        
        # Fitness penalties (matching Go backend)
        self.PENALTY_NO_LUNCH = 5.0
        self.PENALTY_LATE_CLASS = 3.0
        self.PENALTY_EXCESSIVE_DAILY_HOURS = 2.0
        self.PENALTY_GAP = 1.0
        self.PENALTY_UNEVEN_WORKLOAD = 1.5
        
        # Thresholds
        self.MAX_DAILY_HOURS = 8  # Maximum hours per day
        self.MAX_TOTAL_HOURS = 40  # Maximum hours per week
    
    def calculate_fitness(self, week_schedule: List[List[List[int]]]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate fitness score for a schedule.
        
        Args:
            week_schedule: 6 days x 24 slots x 3 attributes [subject_id, instructor_id, room_id]
        
        Returns:
            (fitness_score, details_dict)
        """
        fitness = 0.0
        details = {
            'violations': [],
            'instructor_workloads': {},
            'daily_violations': []
        }
        
        # Collect all instructor schedules
        instructor_schedules = self._get_instructor_schedules(week_schedule)
        
        # Calculate fitness for each instructor
        for instructor_id, schedule in instructor_schedules.items():
            if instructor_id == 0:  # Skip empty slots
                continue
            
            instructor_fitness, instructor_details = self._calculate_instructor_fitness(
                instructor_id, schedule
            )
            
            fitness += instructor_fitness
            details['violations'].extend(instructor_details['violations'])
            details['instructor_workloads'][instructor_id] = instructor_details['workload']
            details['daily_violations'].extend(instructor_details['daily_violations'])
        
        details['total_fitness'] = fitness
        details['num_instructors'] = len([i for i in instructor_schedules.keys() if i != 0])
        
        return fitness, details
    
    def _get_instructor_schedules(self, week_schedule: List[List[List[int]]]) -> Dict[int, List[List[int]]]:
        """
        Extract each instructor's schedule from the week schedule.
        
        Returns:
            {instructor_id: [[day, slot, subject_id, room_id], ...]}
        """
        instructor_schedules = {}
        
        for day_idx, day_schedule in enumerate(week_schedule):
            for slot_idx, slot in enumerate(day_schedule):
                subject_id, instructor_id, room_id = slot
                
                if instructor_id not in instructor_schedules:
                    instructor_schedules[instructor_id] = []
                
                if subject_id > 0:  # Only count actual classes
                    instructor_schedules[instructor_id].append([
                        day_idx, slot_idx, subject_id, room_id
                    ])
        
        return instructor_schedules
    
    def _calculate_instructor_fitness(self, instructor_id: int, schedule: List[List[int]]) -> Tuple[float, Dict]:
        """
        Calculate fitness for a single instructor's schedule.
        """
        fitness = 0.0
        details = {
            'violations': [],
            'workload': {
                'total_hours': 0,
                'daily_hours': [0] * self.DAYS_PER_WEEK,
                'has_lunch_breaks': [False] * self.DAYS_PER_WEEK,
                'late_classes': 0,
                'gaps': 0
            },
            'daily_violations': []
        }
        
        # Organize by day
        daily_schedules = [[] for _ in range(self.DAYS_PER_WEEK)]
        for day, slot, subject, room in schedule:
            daily_schedules[day].append((slot, subject, room))
        
        # Check each day
        for day_idx, day_classes in enumerate(daily_schedules):
            if not day_classes:
                continue
            
            # Sort by time slot
            day_classes.sort(key=lambda x: x[0])
            slots = [slot for slot, _, _ in day_classes]
            
            # Check lunch break
            has_lunch = self._check_lunch_break(slots)
            details['workload']['has_lunch_breaks'][day_idx] = has_lunch
            if not has_lunch:
                fitness += self.PENALTY_NO_LUNCH
                details['violations'].append({
                    'type': 'no_lunch_break',
                    'instructor': instructor_id,
                    'day': day_idx,
                    'penalty': self.PENALTY_NO_LUNCH
                })
                details['daily_violations'].append(f"Day {day_idx}: No lunch break")
            
            # Check late classes
            late_count = sum(1 for slot in slots if slot >= self.LATE_CLASS_SLOT)
            details['workload']['late_classes'] += late_count
            if late_count > 0:
                penalty = self.PENALTY_LATE_CLASS * late_count
                fitness += penalty
                details['violations'].append({
                    'type': 'late_class',
                    'instructor': instructor_id,
                    'day': day_idx,
                    'count': late_count,
                    'penalty': penalty
                })
                details['daily_violations'].append(f"Day {day_idx}: {late_count} late class(es)")
            
            # Calculate daily hours (30 minutes per slot)
            daily_hours = len(slots) * 0.5
            details['workload']['daily_hours'][day_idx] = daily_hours
            details['workload']['total_hours'] += daily_hours
            
            # Check excessive daily hours
            if daily_hours > self.MAX_DAILY_HOURS:
                excess = daily_hours - self.MAX_DAILY_HOURS
                penalty = self.PENALTY_EXCESSIVE_DAILY_HOURS * excess
                fitness += penalty
                details['violations'].append({
                    'type': 'excessive_daily_hours',
                    'instructor': instructor_id,
                    'day': day_idx,
                    'hours': daily_hours,
                    'excess': excess,
                    'penalty': penalty
                })
                details['daily_violations'].append(f"Day {day_idx}: {daily_hours} hours (excess: {excess})")
            
            # Check gaps between classes
            gaps = self._count_gaps(slots)
            details['workload']['gaps'] += gaps
            if gaps > 0:
                penalty = self.PENALTY_GAP * gaps
                fitness += penalty
                details['violations'].append({
                    'type': 'gaps_between_classes',
                    'instructor': instructor_id,
                    'day': day_idx,
                    'count': gaps,
                    'penalty': penalty
                })
        
        # Check workload distribution across week
        daily_hours = details['workload']['daily_hours']
        if any(h > 0 for h in daily_hours):
            working_days = [h for h in daily_hours if h > 0]
            avg_hours = np.mean(working_days)
            std_hours = np.std(working_days)
            
            # Penalty for uneven distribution
            if std_hours > 2.0:  # More than 2 hours standard deviation
                penalty = self.PENALTY_UNEVEN_WORKLOAD * std_hours
                fitness += penalty
                details['violations'].append({
                    'type': 'uneven_workload',
                    'instructor': instructor_id,
                    'std_deviation': std_hours,
                    'penalty': penalty
                })
        
        return fitness, details
    
    def _check_lunch_break(self, slots: List[int]) -> bool:
        """
        Check if there's a lunch break (no classes during 11 AM - 1 PM).
        """
        lunch_slots = set(range(self.LUNCH_START_SLOT, self.LUNCH_END_SLOT))
        class_slots = set(slots)
        
        # If any class during lunch time, no proper lunch break
        return len(lunch_slots & class_slots) == 0
    
    def _count_gaps(self, slots: List[int]) -> int:
        """
        Count gaps (empty slots) between classes.
        """
        if len(slots) <= 1:
            return 0
        
        slots = sorted(slots)
        gaps = 0
        
        for i in range(len(slots) - 1):
            gap_size = slots[i + 1] - slots[i] - 1
            if gap_size > 0:
                gaps += gap_size
        
        return gaps
    
    def print_schedule_analysis(self, details: Dict):
        """
        Print a human-readable analysis of the schedule.
        """
        print("\n" + "="*70)
        print("SCHEDULE FITNESS ANALYSIS")
        print("="*70)
        
        print(f"\n📊 Overall Fitness Score: {details['total_fitness']:.2f}")
        print(f"👥 Number of Instructors: {details['num_instructors']}")
        
        if details['violations']:
            print(f"\n⚠️  Total Violations: {len(details['violations'])}")
            
            # Group by type
            violation_types = {}
            for v in details['violations']:
                vtype = v['type']
                if vtype not in violation_types:
                    violation_types[vtype] = []
                violation_types[vtype].append(v)
            
            for vtype, violations in violation_types.items():
                count = len(violations)
                total_penalty = sum(v['penalty'] for v in violations)
                print(f"   • {vtype.replace('_', ' ').title()}: {count} cases (penalty: {total_penalty:.2f})")
        else:
            print("\n✓ No violations found!")
        
        print("\n" + "="*70)


def load_schedules_json(file_path: str) -> List[Dict]:
    """
    Load schedules from JSON file.
    Handles multiple formats.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON formats
    if isinstance(data, list):
        schedules = data
    elif 'schedules' in data:
        schedules = data['schedules']
    elif 'schedule' in data:
        schedules = [data]
    else:
        raise ValueError(f"Unknown JSON format. Expected list or dict with 'schedules' key.")
    
    return schedules


def calculate_and_save(input_file: str, output_file: str = None, verbose: bool = True):
    """
    Calculate fitness scores for all schedules in input file and save to output file.
    """
    print(f"\n📂 Loading schedules from: {input_file}")
    
    # Load schedules
    schedules = load_schedules_json(input_file)
    print(f"✓ Loaded {len(schedules)} schedule(s)")
    
    # Initialize calculator
    calculator = FitnessCalculator()
    
    # Calculate fitness for each schedule
    print("\n🧮 Calculating fitness scores...")
    
    results = []
    for idx, schedule_data in enumerate(schedules):
        # Extract week schedule
        if 'week_schedule' in schedule_data:
            week_schedule = schedule_data['week_schedule']
        elif 'schedule' in schedule_data and 'week_schedule' in schedule_data['schedule']:
            week_schedule = schedule_data['schedule']['week_schedule']
        else:
            print(f"   ⚠️  Schedule {idx}: No week_schedule found, skipping")
            continue
        
        # Validate format
        if len(week_schedule) != 6:
            print(f"   ⚠️  Schedule {idx}: Invalid format (not 6 days), skipping")
            continue
        
        # Calculate fitness
        fitness, details = calculator.calculate_fitness(week_schedule)
        
        # Create result entry
        result = {
            'id': schedule_data.get('id', f'schedule_{idx}'),
            'fitness': fitness,
            'week_schedule': week_schedule,
            'metadata': {
                'source': 'historical_manual_schedule',
                'num_instructors': details['num_instructors'],
                'num_violations': len(details['violations']),
                'calculated_by': 'calculate_fitness_for_historical.py'
            }
        }
        
        # Include original metadata if present
        if 'metadata' in schedule_data:
            result['metadata'].update(schedule_data['metadata'])
        
        results.append(result)
        
        if verbose:
            print(f"   ✓ Schedule {idx} (ID: {result['id']}): Fitness = {fitness:.2f}")
            if fitness > 50:
                print(f"      ⚠️  High fitness score indicates many violations")
    
    # Calculate statistics
    if results:
        fitnesses = [r['fitness'] for r in results]
        print(f"\n📊 Statistics:")
        print(f"   • Total schedules: {len(results)}")
        print(f"   • Fitness range: {min(fitnesses):.2f} - {max(fitnesses):.2f}")
        print(f"   • Average fitness: {np.mean(fitnesses):.2f}")
        print(f"   • Median fitness: {np.median(fitnesses):.2f}")
        print(f"   • Std deviation: {np.std(fitnesses):.2f}")
        
        # Quality assessment
        excellent = sum(1 for f in fitnesses if f < 20)
        good = sum(1 for f in fitnesses if 20 <= f < 40)
        fair = sum(1 for f in fitnesses if 40 <= f < 60)
        poor = sum(1 for f in fitnesses if f >= 60)
        
        print(f"\n🎯 Quality Distribution:")
        print(f"   • Excellent (< 20): {excellent} ({excellent/len(results)*100:.1f}%)")
        print(f"   • Good (20-40): {good} ({good/len(results)*100:.1f}%)")
        print(f"   • Fair (40-60): {fair} ({fair/len(results)*100:.1f}%)")
        print(f"   • Poor (>= 60): {poor} ({poor/len(results)*100:.1f}%)")
    
    # Save results
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_with_fitness.json")
    
    output_data = {
        'schedules': results,
        'metadata': {
            'source_file': input_file,
            'total_schedules': len(results),
            'calculated_date': '2026-03-06',
            'fitness_calculator': 'FitnessCalculator v1.0'
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Saved {len(results)} schedules with fitness scores to: {output_file}")
    
    return results


def main():
    """
    Command-line interface for fitness calculation.
    """
    if len(sys.argv) < 2:
        print("Usage: python calculate_fitness_for_historical.py <input_json> [output_json]")
        print("\nExample:")
        print("  python calculate_fitness_for_historical.py manual_schedules.json")
        print("  python calculate_fitness_for_historical.py manual_schedules.json with_fitness.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(input_file).exists():
        print(f"❌ Error: File not found: {input_file}")
        sys.exit(1)
    
    try:
        results = calculate_and_save(input_file, output_file, verbose=True)
        
        print("\n" + "="*70)
        print("✅ SUCCESS!")
        print("="*70)
        print("\nNext steps:")
        print("1. Review the fitness scores - manual schedules should have low scores)")
        print("2. Import into training data:")
        print(f"   python import_existing_data.py {output_file or input_file.replace('.json', '_with_fitness.json')}")
        print("3. Train the model:")
        print("   python train_fitness_predictor.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
