"""
Training script for the Fitness Predictor Model
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import config
from models import FitnessPredictorModel
from feature_extraction import FitnessFeatureExtractor
import os
import json
from pathlib import Path


class FitnessPredictorTrainer:
    """
    Trainer class for Fitness Predictor Model
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize trainer
        
        Args:
            data_path: Path to training data file
        """
        self.data_path = data_path or config.DATA_DIR / "training_data.json"
        self.feature_extractor = FitnessFeatureExtractor()
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.history = None
        
        # Set random seeds for reproducibility
        np.random.seed(config.RANDOM_SEED)
        tf.random.set_seed(config.RANDOM_SEED)
    
    def load_data(self) -> tuple:
        """
        Load training data from file
        
        Returns:
            Tuple of (X, y) data
        """
        print(f"Loading data from: {self.data_path}")
        
        if not os.path.exists(self.data_path):
            print(f"WARNING: Data file not found at {self.data_path}")
            print("Generating synthetic data for demonstration...")
            return self._generate_synthetic_data()
        
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        X = []
        y = []
        
        for sample in data:
            features = self.feature_extractor.extract(sample['schedule'])
            X.append(features)
            y.append(sample['fitness'])
        
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        print(f"Loaded {len(X)} samples")
        print(f"Feature shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        return X, y
    
    def _generate_synthetic_data(self, n_samples: int = 1000) -> tuple:
        """
        Generate synthetic training data for testing
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of (X, y)
        """
        print(f"Generating {n_samples} synthetic training samples...")
        
        X = []
        y = []
        
        for _ in range(n_samples):
            # Generate random schedule
            schedule_data = {
                'week_schedule': np.random.randint(0, 10, (6, 24, 3)).tolist()
            }
            
            features = self.feature_extractor.extract(schedule_data)
            X.append(features)
            
            # Generate synthetic fitness (based on some features)
            # Good schedules have: fewer gaps, lunch breaks, no late classes
            fitness = 0.0
            
            # Penalize for gaps
            gaps = features[18]  # Total gaps
            fitness -= gaps * 2.0
            
            # Reward for lunch breaks
            lunch_breaks = sum(features[6:12])  # Lunch break indicators
            fitness += lunch_breaks * 8.0
            
            # Penalize late classes
            late_classes = sum(features[12:18])  # After 5pm classes
            fitness -= late_classes * 4.0
            
            # Add some noise
            fitness += np.random.normal(0, 5)
            
            y.append(fitness)
        
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        print(f"Generated {n_samples} samples")
        print(f"Fitness range: [{y.min():.2f}, {y.max():.2f}]")
        
        return X, y
    
    def preprocess_data(self, X, y):
        """
        Preprocess and split data
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Training, validation, and test sets
        """
        print("\nPreprocessing data...")
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            test_size=(1 - config.TRAIN_RATIO),
            random_state=config.RANDOM_SEED
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(config.TEST_RATIO / (config.TEST_RATIO + config.VALIDATION_RATIO)),
            random_state=config.RANDOM_SEED
        )
        
        # Normalize features
        X_train = self.scaler_X.fit_transform(X_train)
        X_val = self.scaler_X.transform(X_val)
        X_test = self.scaler_X.transform(X_test)
        
        # Normalize targets
        y_train = self.scaler_y.fit_transform(y_train)
        y_val = self.scaler_y.transform(y_val)
        y_test = self.scaler_y.transform(y_test)
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Save scalers
        joblib.dump(self.scaler_X, config.FEATURE_SCALER_PATH)
        joblib.dump(self.scaler_y, config.FITNESS_SCALER_PATH)
        print(f"Scalers saved to {config.MODELS_DIR}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def build_model(self, input_dim: int):
        """
        Build the model
        
        Args:
            input_dim: Number of input features
        """
        print("\nBuilding model...")
        self.model = FitnessPredictorModel.build(input_dim)
        self.model.summary()
    
    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        print("\nStarting training...")
        
        cfg = config.FITNESS_PREDICTOR_CONFIG
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=cfg['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                str(config.FITNESS_PREDICTOR_PATH),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir=str(config.LOGS_DIR / 'fitness_predictor'),
                histogram_freq=1
            )
        ]
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=cfg['epochs'],
            batch_size=cfg['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set
        
        Args:
            X_test: Test features
            y_test: Test targets
        """
        print("\nEvaluating model on test set...")
        
        results = self.model.evaluate(X_test, y_test, verbose=1)
        
        print("\nTest Results:")
        for metric_name, metric_value in zip(self.model.metrics_names, results):
            print(f"  {metric_name}: {metric_value:.4f}")
        
        # Make predictions
        y_pred = self.model.predict(X_test, verbose=0)
        
        # Calculate additional metrics
        from sklearn.metrics import r2_score, mean_absolute_percentage_error
        
        y_test_original = self.scaler_y.inverse_transform(y_test)
        y_pred_original = self.scaler_y.inverse_transform(y_pred)
        
        r2 = r2_score(y_test_original, y_pred_original)
        mape = mean_absolute_percentage_error(y_test_original, y_pred_original)
        
        print(f"\nAdditional Metrics:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAPE: {mape:.4f}%")
        
        return results
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE
        axes[0, 1].plot(self.history.history['mae'], label='Train MAE')
        axes[0, 1].plot(self.history.history['val_mae'], label='Val MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # MSE
        axes[1, 0].plot(self.history.history['mse'], label='Train MSE')
        axes[1, 0].plot(self.history.history['val_mse'], label='Val MSE')
        axes[1, 0].set_title('Mean Squared Error')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # RMSE
        axes[1, 1].plot(self.history.history['rmse'], label='Train RMSE')
        axes[1, 1].plot(self.history.history['val_rmse'], label='Val RMSE')
        axes[1, 1].set_title('Root Mean Squared Error')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = config.LOGS_DIR / 'fitness_predictor_training.png'
        plt.savefig(plot_path, dpi=300)
        print(f"\nTraining plots saved to: {plot_path}")
        plt.show()
    
    def run_full_training_pipeline(self):
        """
        Run the complete training pipeline
        """
        print("=" * 70)
        print("FITNESS PREDICTOR MODEL - TRAINING PIPELINE")
        print("=" * 70)
        
        # Load data
        X, y = self.load_data()
        
        # Preprocess
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_data(X, y)
        
        # Build model
        self.build_model(X_train.shape[1])
        
        # Train
        self.train(X_train, y_train, X_val, y_val)
        
        # Evaluate
        self.evaluate(X_test, y_test)
        
        # Plot
        self.plot_training_history()
        
        print("\n" + "=" * 70)
        print("TRAINING PIPELINE COMPLETED!")
        print("=" * 70)
        print(f"\nModel saved to: {config.FITNESS_PREDICTOR_PATH}")
        print(f"Scalers saved to: {config.MODELS_DIR}")
        print(f"Logs saved to: {config.LOGS_DIR}")


if __name__ == "__main__":
    trainer = FitnessPredictorTrainer()
    trainer.run_full_training_pipeline()
