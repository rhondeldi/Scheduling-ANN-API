"""
Data collection script to gather training data from GA runs
This script should be integrated with the Go backend to collect schedule data
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import src.config as config


class DataCollector:
    """
    Collects training data from GA executions
    """
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize data collector
        
        Args:
            output_dir: Directory to save collected data
        """
        self.output_dir = output_dir or config.DATA_DIR
        self.output_dir.mkdir(exist_ok=True)
        
        self.training_data = []
        self.constraint_data = []
        self.crossover_data = []
        self.mutation_data = []
    
    def collect_fitness_sample(self, 
                               schedule: List[List[List[int]]],
                               fitness: float,
                               metadata: Dict[str, Any] = None):
        """
        Collect a sample for fitness prediction training
        
        Args:
            schedule: Week schedule (6 days x 24 slots x 3 attributes)
            fitness: Fitness score
            metadata: Additional information
        """
        sample = {
            'schedule': {
                'week_schedule': schedule
            },
            'fitness': fitness,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.training_data.append(sample)
    
    def collect_constraint_sample(self,
                                  schedule: List[List[List[int]]],
                                  violations: Dict[str, bool],
                                  metadata: Dict[str, Any] = None):
        """
        Collect a sample for constraint classification training
        
        Args:
            schedule: Week schedule
            violations: Dictionary of constraint violations
            metadata: Additional information
        """
        sample = {
            'schedule': {
                'week_schedule': schedule
            },
            'violations': violations,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.constraint_data.append(sample)
    
    def collect_crossover_sample(self,
                                parent1: List[List[List[int]]],
                                parent2: List[List[List[int]]],
                                crossover_point: int,
                                offspring_fitness: float,
                                metadata: Dict[str, Any] = None):
        """
        Collect a sample for crossover recommendation training
        
        Args:
            parent1: First parent schedule
            parent2: Second parent schedule
            crossover_point: Point where crossover occurred
            offspring_fitness: Fitness of offspring
            metadata: Additional information
        """
        sample = {
            'parent1': {'week_schedule': parent1},
            'parent2': {'week_schedule': parent2},
            'crossover_point': crossover_point,
            'offspring_fitness': offspring_fitness,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.crossover_data.append(sample)
    
    def collect_mutation_sample(self,
                               original_schedule: List[List[List[int]]],
                               mutated_schedule: List[List[List[int]]],
                               original_fitness: float,
                               mutated_fitness: float,
                               mutation_info: Dict[str, Any],
                               metadata: Dict[str, Any] = None):
        """
        Collect a sample for mutation impact prediction
        
        Args:
            original_schedule: Schedule before mutation
            mutated_schedule: Schedule after mutation
            original_fitness: Fitness before mutation
            mutated_fitness: Fitness after mutation
            mutation_info: Information about the mutation
            metadata: Additional information
        """
        # Determine impact
        if mutated_fitness > original_fitness + 1.0:
            impact = 'improve'
        elif mutated_fitness < original_fitness - 1.0:
            impact = 'worsen'
        else:
            impact = 'neutral'
        
        sample = {
            'original_schedule': {'week_schedule': original_schedule},
            'mutated_schedule': {'week_schedule': mutated_schedule},
            'original_fitness': original_fitness,
            'mutated_fitness': mutated_fitness,
            'impact': impact,
            'mutation_info': mutation_info,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.mutation_data.append(sample)
    
    def save_data(self, dataset_name: str = None):
        """
        Save collected data to files
        
        Args:
            dataset_name: Optional name for the dataset
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = dataset_name or timestamp
        
        # Save fitness data
        if self.training_data:
            fitness_path = self.output_dir / f"training_data_{dataset_name}.json"
            with open(fitness_path, 'w') as f:
                json.dump(self.training_data, f, indent=2)
            print(f"Saved {len(self.training_data)} fitness samples to {fitness_path}")
        
        # Save constraint data
        if self.constraint_data:
            constraint_path = self.output_dir / f"constraint_data_{dataset_name}.json"
            with open(constraint_path, 'w') as f:
                json.dump(self.constraint_data, f, indent=2)
            print(f"Saved {len(self.constraint_data)} constraint samples to {constraint_path}")
        
        # Save crossover data
        if self.crossover_data:
            crossover_path = self.output_dir / f"crossover_data_{dataset_name}.json"
            with open(crossover_path, 'w') as f:
                json.dump(self.crossover_data, f, indent=2)
            print(f"Saved {len(self.crossover_data)} crossover samples to {crossover_path}")
        
        # Save mutation data
        if self.mutation_data:
            mutation_path = self.output_dir / f"mutation_data_{dataset_name}.json"
            with open(mutation_path, 'w') as f:
                json.dump(self.mutation_data, f, indent=2)
            print(f"Saved {len(self.mutation_data)} mutation samples to {mutation_path}")
    
    def generate_synthetic_dataset(self, n_samples: int = 1000):
        """
        Generate synthetic training data for testing
        
        Args:
            n_samples: Number of samples to generate
        """
        print(f"Generating {n_samples} synthetic samples...")
        
        for i in range(n_samples):
            # Generate random schedule
            schedule = np.random.randint(0, 10, (6, 24, 3)).tolist()
            
            # Generate synthetic fitness
            fitness = np.random.normal(0, 20)
            
            # Collect
            self.collect_fitness_sample(
                schedule=schedule,
                fitness=fitness,
                metadata={'synthetic': True, 'sample_id': i}
            )
            
            # Generate constraint violations with some probability
            violations = {
                'instructor_conflict': np.random.random() > 0.8,
                'room_conflict': np.random.random() > 0.8,
                'no_lunch_break': np.random.random() > 0.5,
                'late_classes': np.random.random() > 0.6,
                'excessive_hours': np.random.random() > 0.7,
                'saturday_overload': np.random.random() > 0.8,
                'resource_unavailable': np.random.random() > 0.9,
                'curriculum_conflict': np.random.random() > 0.85,
                'room_capacity': np.random.random() > 0.9,
                'instructor_availability': np.random.random() > 0.85
            }
            
            self.collect_constraint_sample(
                schedule=schedule,
                violations=violations,
                metadata={'synthetic': True, 'sample_id': i}
            )
        
        print(f"Generated {n_samples} synthetic samples")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about collected data
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'fitness_samples': len(self.training_data),
            'constraint_samples': len(self.constraint_data),
            'crossover_samples': len(self.crossover_data),
            'mutation_samples': len(self.mutation_data),
            'total_samples': (
                len(self.training_data) + 
                len(self.constraint_data) + 
                len(self.crossover_data) + 
                len(self.mutation_data)
            )
        }
        
        if self.training_data:
            fitnesses = [s['fitness'] for s in self.training_data]
            stats['fitness_stats'] = {
                'min': min(fitnesses),
                'max': max(fitnesses),
                'mean': np.mean(fitnesses),
                'std': np.std(fitnesses)
            }
        
        return stats


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("DATA COLLECTION SCRIPT")
    print("=" * 70)
    
    collector = DataCollector()
    
    # Generate synthetic data for demonstration
    collector.generate_synthetic_dataset(n_samples=1000)
    
    # Print statistics
    stats = collector.get_statistics()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save data
    collector.save_data(dataset_name="synthetic_demo")
    
    print("\n" + "=" * 70)
    print("DATA COLLECTION COMPLETED")
    print("=" * 70)
