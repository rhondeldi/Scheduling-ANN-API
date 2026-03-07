"""
Feature extraction from schedule data for ANN models
"""
import numpy as np
from typing import Dict, List, Any
import src.config as config

class ScheduleFeatureExtractor:
    """
    Extract meaningful features from schedule data for ANN input
    """
    
    def __init__(self):
        self.n_days = config.N_WEEKLY_SCHOOL_DAYS
        self.n_slots = config.N_DAILY_TIME_SLOTS
        
    def extract_features(self, schedule_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract comprehensive features from a schedule
        
        Args:
            schedule_data: Dictionary containing schedule information
                Format: {
                    'week_schedule': [[[subject_id, instructor_id, room_id], ...], ...],
                    'curriculum_info': {...},
                    'resources': {...}
                }
        
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        week_schedule = self._parse_schedule(schedule_data)
        
        # Feature Group 1: Temporal Distribution (12 features)
        features.extend(self._extract_temporal_features(week_schedule))
        
        # Feature Group 2: Constraint-related (12 features)
        features.extend(self._extract_constraint_features(week_schedule))
        
        # Feature Group 3: Resource Utilization (8 features)
        features.extend(self._extract_resource_features(week_schedule, schedule_data))
        
        # Feature Group 4: Distribution Quality (10 features)
        features.extend(self._extract_distribution_features(week_schedule))
        
        # Feature Group 5: Workload Balance (8 features)
        features.extend(self._extract_workload_features(week_schedule, schedule_data))
        
        return np.array(features, dtype=np.float32)
    
    def _parse_schedule(self, schedule_data: Dict) -> np.ndarray:
        """
        Parse schedule data into standardized format
        Returns: (days, time_slots, 3) array
        """
        if isinstance(schedule_data, np.ndarray):
            return schedule_data
            
        week_schedule = schedule_data.get('week_schedule', [])
        
        # Convert to numpy array
        parsed = np.zeros((self.n_days, self.n_slots, 3), dtype=np.int32)
        
        for day_idx, day_schedule in enumerate(week_schedule[:self.n_days]):
            for slot_idx, slot in enumerate(day_schedule[:self.n_slots]):
                if len(slot) >= 3:
                    parsed[day_idx, slot_idx] = slot[:3]
        
        return parsed
    
    def _extract_temporal_features(self, schedule: np.ndarray) -> List[float]:
        """
        Features related to time distribution
        """
        features = []
        
        for day in range(self.n_days):
            day_schedule = schedule[day]
            occupied_slots = np.sum(day_schedule[:, 0] > 0)
            daily_hours = occupied_slots / config.N_HOUR_TIME_SLOTS
            features.append(daily_hours)
        
        # Total weekly hours
        total_hours = sum(features)
        features.append(total_hours)
        
        # Days with classes
        days_with_classes = sum(1 for h in features[:6] if h > 0)
        features.append(days_with_classes)
        
        # Hour distribution variance
        hour_variance = np.var(features[:6])
        features.append(hour_variance)
        
        # Average hours per active day
        avg_hours = total_hours / days_with_classes if days_with_classes > 0 else 0
        features.append(avg_hours)
        
        # Saturday hours (penalized if too long)
        features.append(features[5])  # Saturday is index 5
        
        # Max hours in a single day
        features.append(max(features[:6]))
        
        return features
    
    def _extract_constraint_features(self, schedule: np.ndarray) -> List[float]:
        """
        Features related to scheduling constraints
        """
        features = []
        
        for day in range(self.n_days):
            day_schedule = schedule[day]
            
            # Check lunch break (slots 10-13 represent 12pm-2pm if 7am start)
            lunch_slots = day_schedule[10:14, 0]  # 12pm to 2pm
            has_lunch_break = np.any(lunch_slots == 0)
            features.append(float(has_lunch_break))
            
            # Classes after 5pm (slot 20 onwards if 7am start)
            after_5pm = np.sum(day_schedule[20:, 0] > 0)
            features.append(float(after_5pm))
        
        return features
    
    def _extract_resource_features(self, schedule: np.ndarray, 
                                   schedule_data: Dict) -> List[float]:
        """
        Features related to resource usage (instructors, rooms)
        """
        features = []
        
        # Unique instructors used
        instructor_ids = schedule[:, :, 1].flatten()
        unique_instructors = len(np.unique(instructor_ids[instructor_ids > 0]))
        features.append(float(unique_instructors))
        
        # Unique rooms used
        room_ids = schedule[:, :, 2].flatten()
        unique_rooms = len(np.unique(room_ids[room_ids > 0]))
        features.append(float(unique_rooms))
        
        # Unique subjects
        subject_ids = schedule[:, :, 0].flatten()
        unique_subjects = len(np.unique(subject_ids[subject_ids > 0]))
        features.append(float(unique_subjects))
        
        # Instructor load variance
        instructor_counts = {}
        for iid in instructor_ids:
            if iid > 0:
                instructor_counts[iid] = instructor_counts.get(iid, 0) + 1
        
        if instructor_counts:
            load_variance = np.var(list(instructor_counts.values()))
            max_load = max(instructor_counts.values())
            min_load = min(instructor_counts.values())
            avg_load = np.mean(list(instructor_counts.values()))
        else:
            load_variance = 0
            max_load = 0
            min_load = 0
            avg_load = 0
        
        features.extend([load_variance, max_load, min_load, avg_load])
        
        return features
    
    def _extract_distribution_features(self, schedule: np.ndarray) -> List[float]:
        """
        Features about how classes are distributed
        """
        features = []
        
        for day in range(self.n_days):
            day_schedule = schedule[day, :, 0]  # Just subject IDs
            
            # Count gaps (empty slots between classes)
            occupied = (day_schedule > 0).astype(int)
            if occupied.sum() == 0:
                gaps = 0
            else:
                first_class = np.argmax(occupied)
                last_class = len(occupied) - np.argmax(occupied[::-1]) - 1
                gaps = (last_class - first_class + 1) - occupied.sum()
            
            features.append(float(gaps))
        
        # Total gaps across week
        total_gaps = sum(features)
        features.append(total_gaps)
        
        # Class distribution compactness (lower gaps = more compact)
        avg_gaps_per_day = total_gaps / self.n_days
        features.append(avg_gaps_per_day)
        
        # Longest continuous teaching period
        max_continuous = 0
        for day in range(self.n_days):
            day_schedule = schedule[day, :, 0]
            continuous = 0
            for slot in day_schedule:
                if slot > 0:
                    continuous += 1
                    max_continuous = max(max_continuous, continuous)
                else:
                    continuous = 0
        features.append(float(max_continuous))
        
        return features
    
    def _extract_workload_features(self, schedule: np.ndarray, 
                                   schedule_data: Dict) -> List[float]:
        """
        Features about instructor and student workload patterns
        """
        features = []
        
        # Morning classes (before 12pm - first 10 slots)
        morning_count = np.sum(schedule[:, :10, 0] > 0)
        features.append(float(morning_count))
        
        # Afternoon classes (12pm-5pm - slots 10-20)
        afternoon_count = np.sum(schedule[:, 10:20, 0] > 0)
        features.append(float(afternoon_count))
        
        # Evening classes (after 5pm - slots 20+)
        evening_count = np.sum(schedule[:, 20:, 0] > 0)
        features.append(float(evening_count))
        
        # Ratio metrics
        total_classes = morning_count + afternoon_count + evening_count
        if total_classes > 0:
            morning_ratio = morning_count / total_classes
            afternoon_ratio = afternoon_count / total_classes
            evening_ratio = evening_count / total_classes
        else:
            morning_ratio = afternoon_ratio = evening_ratio = 0
        
        features.extend([morning_ratio, afternoon_ratio, evening_ratio])
        
        # Weekend (Saturday) vs weekday balance
        weekday_hours = np.sum(schedule[:5, :, 0] > 0) / config.N_HOUR_TIME_SLOTS
        saturday_hours = np.sum(schedule[5, :, 0] > 0) / config.N_HOUR_TIME_SLOTS
        
        weekend_ratio = saturday_hours / (weekday_hours + saturday_hours) if weekday_hours + saturday_hours > 0 else 0
        features.append(weekend_ratio)
        
        # Spread factor (how spread out are classes)
        occupied_slots = np.where(schedule[:, :, 0].flatten() > 0)[0]
        if len(occupied_slots) > 1:
            spread = (occupied_slots[-1] - occupied_slots[0]) / len(occupied_slots)
        else:
            spread = 0
        features.append(spread)
        
        return features


class FitnessFeatureExtractor:
    """
    Extract features specifically for fitness prediction
    """
    
    def __init__(self):
        self.schedule_extractor = ScheduleFeatureExtractor()
    
    def extract(self, schedule_data: Dict) -> np.ndarray:
        """
        Extract features for fitness prediction model
        """
        return self.schedule_extractor.extract_features(schedule_data)


class ConstraintFeatureExtractor:
    """
    Extract features for constraint violation prediction
    """
    
    def __init__(self):
        self.schedule_extractor = ScheduleFeatureExtractor()
    
    def extract(self, schedule_data: Dict) -> np.ndarray:
        """
        Extract features for constraint checking
        Includes same features as fitness + additional constraint-specific ones
        """
        base_features = self.schedule_extractor.extract_features(schedule_data)
        
        # Add constraint-specific features
        schedule = self.schedule_extractor._parse_schedule(schedule_data)
        constraint_features = []
        
        # Check for instructor conflicts (same instructor, same time, different places)
        for time_slot_idx in range(config.N_DAILY_TIME_SLOTS):
            for day in range(config.N_WEEKLY_SCHOOL_DAYS):
                # This would require more context about the full university schedule
                # For now, placeholder
                pass
        
        # Room capacity violations (would need room capacity data)
        # Instructor availability violations (would need availability data)
        
        return base_features  # Extend with constraint features as needed


def create_feature_extractors():
    """
    Factory function to create all feature extractors
    """
    return {
        'fitness': FitnessFeatureExtractor(),
        'constraint': ConstraintFeatureExtractor(),
        'general': ScheduleFeatureExtractor()
    }


# Example usage
if __name__ == "__main__":
    # Example schedule data
    example_schedule = {
        'week_schedule': np.random.randint(0, 10, (6, 24, 3)).tolist()
    }
    
    extractor = ScheduleFeatureExtractor()
    features = extractor.extract_features(example_schedule)
    
    print(f"Extracted {len(features)} features:")
    print(features)
    print(f"Feature shape: {features.shape}")
