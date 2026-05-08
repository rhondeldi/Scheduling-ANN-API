"""
Configuration settings for the ANN models.
"""
import os
from pathlib import Path


# ── Project paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
LOGS_DIR     = PROJECT_ROOT / "logs"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ── Schedule constants (mirrors Go backend) ────────────────────────────────────
N_WEEKLY_SCHOOL_DAYS = 6   # Monday–Saturday
N_DAILY_TIME_SLOTS   = 24  # 7 AM–7 PM in 30-min slots
N_HOUR_TIME_SLOTS    = 2   # 2 slots = 1 hour

TOTAL_TIME_SLOTS  = N_WEEKLY_SCHOOL_DAYS * N_DAILY_TIME_SLOTS  # 144
ATTRIBUTES_PER_SLOT = 3                                         # subject, instructor, room
FLAT_SCHEDULE_SIZE  = TOTAL_TIME_SLOTS * ATTRIBUTES_PER_SLOT   # 432

# ── Model hyperparameters ──────────────────────────────────────────────────────
# Fitness model uses the 48-feature extractor.
FITNESS_PREDICTOR_CONFIG = {
    'input_dim':               48,
    'hidden_layers':           [256, 128, 64],
    'dropout_rate':            0.2,
    'learning_rate':           0.001,
    'batch_size':              64,
    'epochs':                  300,
    'early_stopping_patience': 30,
}

CONSTRAINT_CLASSIFIER_CONFIG = {
    'input_dim':               48,
    'hidden_layers':           [256, 128, 64],
    'dropout_rates':           [0.30, 0.20, 0.10],
    'num_constraint_types':    10,
    'learning_rate':           0.001,
    'batch_size':              32,
    'epochs':                  200,
    'early_stopping_patience': 20,
}

CROSSOVER_RECOMMENDER_CONFIG = {
    'lstm_units':   128,
    'dense_units':  64,
    'dropout_rate': 0.20,
    'learning_rate': 0.001,
    'batch_size':   16,
    'epochs':       150,
}

MUTATION_PREDICTOR_CONFIG = {
    'input_dim':      60,
    'hidden_layers':  [128, 64],
    'output_classes': 3,
    'dropout_rate':   0.20,
    'learning_rate':  0.001,
    'batch_size':     32,
    'epochs':         150,
}

# ── Training split ─────────────────────────────────────────────────────────────
TRAIN_RATIO      = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO       = 0.15

# ── API ────────────────────────────────────────────────────────────────────────
API_HOST    = os.getenv("ANN_API_HOST",    "0.0.0.0")
API_PORT    = int(os.getenv("ANN_API_PORT", "8000"))
API_WORKERS = int(os.getenv("ANN_API_WORKERS", "1"))

GO_BACKEND_URL = os.getenv("GO_BACKEND_URL", "http://localhost:3000")

# ── Model / scaler paths ───────────────────────────────────────────────────────
FITNESS_PREDICTOR_PATH    = MODELS_DIR / "fitness_predictor.keras"
CONSTRAINT_CLASSIFIER_PATH = MODELS_DIR / "constraint_classifier.keras"
CROSSOVER_RECOMMENDER_PATH = MODELS_DIR / "crossover_recommender.keras"
MUTATION_PREDICTOR_PATH    = MODELS_DIR / "mutation_predictor.keras"

FEATURE_SCALER_PATH = MODELS_DIR / "feature_scaler.joblib"
FITNESS_SCALER_PATH = MODELS_DIR / "fitness_scaler.joblib"
CONSTRAINT_SCALER_PATH = MODELS_DIR / "constraint_scaler.joblib"
MUTATION_SCALER_PATH = MODELS_DIR / "mutation_scaler.joblib"

# ── Misc ───────────────────────────────────────────────────────────────────────
LOG_LEVEL   = os.getenv("LOG_LEVEL", "INFO")
RANDOM_SEED = 42

print("Configuration loaded:")
print(f"  Data dir    : {DATA_DIR}")
print(f"  Models dir  : {MODELS_DIR}")
print(f"  API         : {API_HOST}:{API_PORT}")
print(f"  input_dim   : {FITNESS_PREDICTOR_CONFIG['input_dim']}")
