"""
FastAPI service for serving ANN models
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
import config
from feature_extraction import create_feature_extractors
import os
from datetime import datetime


# Initialize FastAPI app
app = FastAPI(
    title="Scheduling ANN API",
    description="API for ANN models assisting the Genetic Algorithm scheduling system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and extractors
models = {}
scalers = {}
feature_extractors = {}


# Pydantic models for request/response
class ScheduleData(BaseModel):
    """Schedule data format"""
    week_schedule: List[List[List[int]]]  # 6 days x 24 slots x 3 attributes
    curriculum_info: Optional[Dict[str, Any]] = None
    resources: Optional[Dict[str, Any]] = None


class FitnessPredictionRequest(BaseModel):
    """Request for fitness prediction"""
    schedule: ScheduleData


class FitnessPredictionResponse(BaseModel):
    """Response for fitness prediction"""
    predicted_fitness: float
    confidence: Optional[float] = None
    processing_time_ms: float


class ConstraintCheckRequest(BaseModel):
    """Request for constraint checking"""
    schedule: ScheduleData


class ConstraintCheckResponse(BaseModel):
    """Response for constraint checking"""
    violations: Dict[str, bool]
    violation_scores: Dict[str, float]
    processing_time_ms: float


class CrossoverRecommendationRequest(BaseModel):
    """Request for crossover recommendation"""
    parent1: ScheduleData
    parent2: ScheduleData
    parent1_fitness: float
    parent2_fitness: float


class CrossoverRecommendationResponse(BaseModel):
    """Response for crossover recommendation"""
    recommended_points: List[int]
    probabilities: List[float]
    processing_time_ms: float


class MutationPredictionRequest(BaseModel):
    """Request for mutation prediction"""
    current_schedule: ScheduleData
    proposed_mutation: Dict[str, Any]
    current_fitness: float


class MutationPredictionResponse(BaseModel):
    """Response for mutation prediction"""
    prediction: str  # 'improve', 'neutral', or 'worsen'
    confidence: float
    probabilities: Dict[str, float]
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: Dict[str, bool]
    timestamp: str


# Startup event - load models
@app.on_event("startup")
async def startup_event():
    """Load all models and scalers on startup"""
    print("Loading models and scalers...")
    
    global models, scalers, feature_extractors
    
    # Create feature extractors
    feature_extractors = create_feature_extractors()
    
    # Load fitness predictor
    if os.path.exists(config.FITNESS_PREDICTOR_PATH):
        try:
            models['fitness'] = keras.models.load_model(config.FITNESS_PREDICTOR_PATH)
            print(f"✓ Loaded fitness predictor from {config.FITNESS_PREDICTOR_PATH}")
        except Exception as e:
            print(f"✗ Failed to load fitness predictor: {e}")
            models['fitness'] = None
    else:
        print(f"✗ Fitness predictor not found at {config.FITNESS_PREDICTOR_PATH}")
        models['fitness'] = None
    
    # Load constraint classifier
    if os.path.exists(config.CONSTRAINT_CLASSIFIER_PATH):
        try:
            models['constraint'] = keras.models.load_model(config.CONSTRAINT_CLASSIFIER_PATH)
            print(f"✓ Loaded constraint classifier from {config.CONSTRAINT_CLASSIFIER_PATH}")
        except Exception as e:
            print(f"✗ Failed to load constraint classifier: {e}")
            models['constraint'] = None
    else:
        print(f"✗ Constraint classifier not found at {config.CONSTRAINT_CLASSIFIER_PATH}")
        models['constraint'] = None
    
    # Load crossover recommender
    if os.path.exists(config.CROSSOVER_RECOMMENDER_PATH):
        try:
            models['crossover'] = keras.models.load_model(config.CROSSOVER_RECOMMENDER_PATH)
            print(f"✓ Loaded crossover recommender from {config.CROSSOVER_RECOMMENDER_PATH}")
        except Exception as e:
            print(f"✗ Failed to load crossover recommender: {e}")
            models['crossover'] = None
    else:
        print(f"✗ Crossover recommender not found at {config.CROSSOVER_RECOMMENDER_PATH}")
        models['crossover'] = None
    
    # Load mutation predictor
    if os.path.exists(config.MUTATION_PREDICTOR_PATH):
        try:
            models['mutation'] = keras.models.load_model(config.MUTATION_PREDICTOR_PATH)
            print(f"✓ Loaded mutation predictor from {config.MUTATION_PREDICTOR_PATH}")
        except Exception as e:
            print(f"✗ Failed to load mutation predictor: {e}")
            models['mutation'] = None
    else:
        print(f"✗ Mutation predictor not found at {config.MUTATION_PREDICTOR_PATH}")
        models['mutation'] = None
    
    # Load scalers
    if os.path.exists(config.FEATURE_SCALER_PATH):
        try:
            scalers['features'] = joblib.load(config.FEATURE_SCALER_PATH)
            print(f"✓ Loaded feature scaler")
        except Exception as e:
            print(f"✗ Failed to load feature scaler: {e}")
            scalers['features'] = None
    
    if os.path.exists(config.FITNESS_SCALER_PATH):
        try:
            scalers['fitness'] = joblib.load(config.FITNESS_SCALER_PATH)
            print(f"✓ Loaded fitness scaler")
        except Exception as e:
            print(f"✗ Failed to load fitness scaler: {e}")
            scalers['fitness'] = None
    
    print("\nStartup complete!")
    print(f"Models loaded: {sum(1 for m in models.values() if m is not None)}/{len(models)}")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Scheduling ANN API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "fitness": "/predict/fitness",
            "constraints": "/predict/constraints",
            "crossover": "/recommend/crossover",
            "mutation": "/predict/mutation"
        }
    }


# Health check
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "fitness_predictor": models.get('fitness') is not None,
            "constraint_classifier": models.get('constraint') is not None,
            "crossover_recommender": models.get('crossover') is not None,
            "mutation_predictor": models.get('mutation') is not None
        },
        timestamp=datetime.now().isoformat()
    )


# Fitness prediction endpoint
@app.post("/predict/fitness", response_model=FitnessPredictionResponse)
async def predict_fitness(request: FitnessPredictionRequest):
    """
    Predict fitness score for a schedule
    """
    start_time = datetime.now()
    
    if models.get('fitness') is None:
        raise HTTPException(status_code=503, detail="Fitness predictor model not loaded")
    
    if scalers.get('features') is None or scalers.get('fitness') is None:
        raise HTTPException(status_code=503, detail="Scalers not loaded")
    
    try:
        # Extract features
        schedule_dict = request.schedule.dict()
        features = feature_extractors['fitness'].extract(schedule_dict)
        
        # Normalize features
        features_normalized = scalers['features'].transform(features.reshape(1, -1))
        
        # Predict
        prediction_normalized = models['fitness'].predict(features_normalized, verbose=0)
        
        # Denormalize
        prediction = scalers['fitness'].inverse_transform(prediction_normalized)[0][0]
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return FitnessPredictionResponse(
            predicted_fitness=float(prediction),
            confidence=None,  # Can be enhanced with ensemble methods
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Constraint checking endpoint
@app.post("/predict/constraints", response_model=ConstraintCheckResponse)
async def check_constraints(request: ConstraintCheckRequest):
    """
    Check for constraint violations in a schedule
    """
    start_time = datetime.now()
    
    if models.get('constraint') is None:
        raise HTTPException(status_code=503, detail="Constraint classifier model not loaded")
    
    try:
        # Extract features
        schedule_dict = request.schedule.dict()
        features = feature_extractors['constraint'].extract(schedule_dict)
        
        # Normalize if scaler available
        if scalers.get('features') is not None:
            features = scalers['features'].transform(features.reshape(1, -1))
        else:
            features = features.reshape(1, -1)
        
        # Predict
        predictions = models['constraint'].predict(features, verbose=0)[0]
        
        # Map predictions to constraint names
        constraint_names = [
            'instructor_conflict',
            'room_conflict',
            'no_lunch_break',
            'late_classes',
            'excessive_hours',
            'saturday_overload',
            'resource_unavailable',
            'curriculum_conflict',
            'room_capacity',
            'instructor_availability'
        ]
        
        violations = {name: bool(pred > 0.5) for name, pred in zip(constraint_names, predictions)}
        violation_scores = {name: float(pred) for name, pred in zip(constraint_names, predictions)}
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ConstraintCheckResponse(
            violations=violations,
            violation_scores=violation_scores,
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Constraint checking failed: {str(e)}")


# Crossover recommendation endpoint
@app.post("/recommend/crossover", response_model=CrossoverRecommendationResponse)
async def recommend_crossover(request: CrossoverRecommendationRequest):
    """
    Recommend optimal crossover points
    """
    start_time = datetime.now()
    
    if models.get('crossover') is None:
        raise HTTPException(status_code=503, detail="Crossover recommender model not loaded")
    
    try:
        # Convert schedules to sequences
        parent1_seq = np.array(request.parent1.week_schedule).reshape(1, -1, 3)
        parent2_seq = np.array(request.parent2.week_schedule).reshape(1, -1, 3)
        
        # Predict crossover probabilities
        probabilities = models['crossover'].predict([parent1_seq, parent2_seq], verbose=0)[0]
        
        # Get top 5 recommended points
        top_indices = np.argsort(probabilities)[-5:][::-1]
        top_probs = probabilities[top_indices]
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return CrossoverRecommendationResponse(
            recommended_points=top_indices.tolist(),
            probabilities=top_probs.tolist(),
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crossover recommendation failed: {str(e)}")


# Mutation prediction endpoint
@app.post("/predict/mutation", response_model=MutationPredictionResponse)
async def predict_mutation(request: MutationPredictionRequest):
    """
    Predict impact of a proposed mutation
    """
    start_time = datetime.now()
    
    if models.get('mutation') is None:
        raise HTTPException(status_code=503, detail="Mutation predictor model not loaded")
    
    try:
        # Extract features from current schedule
        schedule_dict = request.current_schedule.dict()
        schedule_features = feature_extractors['general'].extract_features(schedule_dict)
        
        # Encode mutation information
        mutation_features = np.array([
            request.current_fitness,
            request.proposed_mutation.get('type', 0),
            request.proposed_mutation.get('position', 0),
            # Add more mutation-specific features
        ])
        
        # Combine features
        combined_features = np.concatenate([schedule_features, mutation_features])
        
        # Pad or truncate to expected input size
        expected_size = config.MUTATION_PREDICTOR_CONFIG['input_dim']
        if len(combined_features) < expected_size:
            combined_features = np.pad(combined_features, (0, expected_size - len(combined_features)))
        else:
            combined_features = combined_features[:expected_size]
        
        # Predict
        predictions = models['mutation'].predict(combined_features.reshape(1, -1), verbose=0)[0]
        
        # Map to class names
        class_names = ['improve', 'neutral', 'worsen']
        predicted_class = class_names[np.argmax(predictions)]
        confidence = float(np.max(predictions))
        
        probabilities = {name: float(prob) for name, prob in zip(class_names, predictions)}
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return MutationPredictionResponse(
            prediction=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mutation prediction failed: {str(e)}")


# Run server
if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("Starting Scheduling ANN API Server")
    print("=" * 70)
    print(f"Host: {config.API_HOST}")
    print(f"Port: {config.API_PORT}")
    print(f"Workers: {config.API_WORKERS}")
    print("=" * 70)
    
    uvicorn.run(
        "api_service:app",
        host=config.API_HOST,
        port=config.API_PORT,
        workers=config.API_WORKERS,
        reload=False
    )
