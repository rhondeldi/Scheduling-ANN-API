"""
Configuration settings for the ANN models
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Schedule constants (from Go backend)
N_WEEKLY_SCHOOL_DAYS = 6  # Monday to Saturday
N_DAILY_TIME_SLOTS = 24   # 7am to 7pm in 30-minute slots
N_HOUR_TIME_SLOTS = 2     # 30-minute slots per hour

# Feature dimensions
TOTAL_TIME_SLOTS = N_WEEKLY_SCHOOL_DAYS * N_DAILY_TIME_SLOTS  # 144
ATTRIBUTES_PER_SLOT = 3  # subject_id, instructor_id, room_id
FLAT_SCHEDULE_SIZE = TOTAL_TIME_SLOTS * ATTRIBUTES_PER_SLOT  # 432

# Model hyperparameters
FITNESS_PREDICTOR_CONFIG = {
    'input_dim': 50,  # Engineered features (not flat)
    'hidden_layers': [128, 64, 32],
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 200,
    'early_stopping_patience': 20
}

CONSTRAINT_CLASSIFIER_CONFIG = {
    'input_dim': 50,
    'hidden_layers': [256, 128, 64],
    'dropout_rates': [0.3, 0.2, 0.1],
    'num_constraint_types': 10,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 200,
    'early_stopping_patience': 15
}

CROSSOVER_RECOMMENDER_CONFIG = {
    'lstm_units': 128,
    'dense_units': 64,
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'batch_size': 16,
    'epochs': 150
}

MUTATION_PREDICTOR_CONFIG = {
    'input_dim': 60,
    'hidden_layers': [128, 64],
    'output_classes': 3,  # Improve, Neutral, Worsen
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 150
}

# Training data split
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# API configuration
API_HOST = os.getenv("ANN_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("ANN_API_PORT", "8000"))
API_WORKERS = int(os.getenv("ANN_API_WORKERS", "4"))

# Go backend configuration
GO_BACKEND_URL = os.getenv("GO_BACKEND_URL", "http://localhost:3000")

# Model paths
FITNESS_PREDICTOR_PATH = MODELS_DIR / "fitness_predictor.h5"
CONSTRAINT_CLASSIFIER_PATH = MODELS_DIR / "constraint_classifier.h5"
CROSSOVER_RECOMMENDER_PATH = MODELS_DIR / "crossover_recommender.h5"
MUTATION_PREDICTOR_PATH = MODELS_DIR / "mutation_predictor.h5"

# Scaler paths (for normalization)
FEATURE_SCALER_PATH = MODELS_DIR / "feature_scaler.joblib"
FITNESS_SCALER_PATH = MODELS_DIR / "fitness_scaler.joblib"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Random seed for reproducibility
RANDOM_SEED = 42

print(f"Configuration loaded:")
print(f"  - Data directory: {DATA_DIR}")
print(f"  - Models directory: {MODELS_DIR}")
print(f"  - API will run on: {API_HOST}:{API_PORT}")
