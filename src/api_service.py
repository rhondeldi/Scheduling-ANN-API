"""
FastAPI service for serving ANN models
"""
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, model_validator
from typing import List, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager
import numpy as np
import joblib
import os
import logging
from datetime import datetime
import json
from tensorflow import keras
import traceback
import socket
import sys

try:
    from src import config
    from src.feature_extraction import create_feature_extractors
except ModuleNotFoundError:
    # Allow direct execution via: python src/api_service.py
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src import config
    from src.feature_extraction import create_feature_extractors


# ── logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("ann_api")


# ── global state — declared before lifespan so the order is unambiguous ───────

models: Dict[str, Any] = {}
scalers: Dict[str, Any] = {}
feature_extractors: Dict[str, Any] = {}
_constraint_model: keras.Model | None = None
_constraint_scaler: Any | None = None
_crossover_model: keras.Model | None = None
_crossover_scaler: Any | None = None
_mutation_model: keras.Model | None = None
_mutation_scaler: Any | None = None



# Required model keys and their human-readable names, in priority order.
_REQUIRED_MODELS = [
    ("fitness",    "fitness_predictor"),
    ("constraint", "constraint_classifier"),
    ("crossover",  "crossover_recommender"),
    ("mutation",   "mutation_predictor"),
]

# ── startup ───────────────────────────────────────────────────────────────────

async def _load_models_and_scalers() -> None:
    """Load all models and scalers. Logs every outcome clearly."""
    logger.info("=" * 60)
    logger.info("STARTUP: loading models and scalers")
    logger.info("=" * 60)

    global models, scalers, feature_extractors

    # Feature extractors
    try:
        feature_extractors = create_feature_extractors()
        logger.info(f"✓ feature extractors created: {list(feature_extractors.keys())}")
    except Exception as exc:
        logger.error(f"✗ failed to create feature extractors: {exc}")
        feature_extractors = {}

    # Models
    model_specs = [
        ("fitness",    config.FITNESS_PREDICTOR_PATH),
        ("constraint", config.CONSTRAINT_CLASSIFIER_PATH),
        ("crossover",  config.CROSSOVER_RECOMMENDER_PATH),
        ("mutation",   config.MUTATION_PREDICTOR_PATH),
    ]
    for key, path in model_specs:
        if not os.path.exists(path):
            logger.warning(f"✗ {key}: file not found at {path}")
            models[key] = None
            continue
        try:
            # compile=False loads weights + architecture without rebuilding the
            # training loss/optimiser.  The constraint classifier was trained
            # with a custom weighted_bce closure that Keras can't deserialise
            # at load time, and we're inference-only here — no need to compile.
            models[key] = keras.models.load_model(path, compile=False)
            logger.info(f"✓ {key}: loaded from {path}")
        except Exception as exc:
            logger.error(f"✗ {key}: load failed — {exc}")
            models[key] = None

    # Scalers
    scaler_specs = [
        ("features",  config.FEATURE_SCALER_PATH),
        ("fitness",   config.FITNESS_SCALER_PATH),
        ("constraint",config.CONSTRAINT_SCALER_PATH),
        ("mutation",  config.MUTATION_SCALER_PATH),
        ("crossover", config.CROSSOVER_SCALER_PATH),
    ]
    for key, path in scaler_specs:
        if not os.path.exists(path):
            logger.warning(f"✗ scaler[{key}]: file not found at {path}")
            scalers[key] = None
            continue
        try:
            scalers[key] = joblib.load(path)
            logger.info(f"✓ scaler[{key}]: loaded from {path}")
        except Exception as exc:
            logger.error(f"✗ scaler[{key}]: load failed — {exc}")
            scalers[key] = None

    # FIX 6: structured final summary so the startup log is unambiguous.
    loaded_models   = [k for k, v in models.items()  if v is not None]
    missing_models  = [k for k, v in models.items()  if v is None]
    loaded_scalers  = [k for k, v in scalers.items() if v is not None]
    missing_scalers = [k for k, v in scalers.items() if v is None]
    extractor_keys  = list(feature_extractors.keys())

    logger.info("")
    logger.info("#" * 70)
    logger.info("# STARTUP REPORT")
    logger.info("#" * 70)

    # Per-model line with explicit OK / MISSING marker so each one stands out.
    logger.info("# MODELS:")
    for key in ("fitness", "constraint", "crossover", "mutation"):
        if models.get(key) is not None:
            logger.info(f"#   [OK]      {key:<11} predictor")
        else:
            logger.error(f"#   [MISSING] {key:<11} predictor — endpoints using it will return 503")

    logger.info("# SCALERS:")
    for key in ("features", "fitness", "constraint", "mutation", "crossover"):
        if scalers.get(key) is not None:
            logger.info(f"#   [OK]      {key:<11} scaler")
        else:
            logger.warning(f"#   [MISSING] {key:<11} scaler")

    logger.info(f"# FEATURE EXTRACTORS: {extractor_keys if extractor_keys else 'NONE LOADED'}")
    if "mutation" not in extractor_keys:
        logger.warning(
            "#   note: no 'mutation' extractor key — /predict/mutation will fall back "
            "to another extractor (fitness > constraint > crossover)"
        )
    logger.info("#" * 70)
    logger.info("")

    # Hard error if the fitness predictor is missing — the Go client's
    # HealthCheck() rejects the API entirely without it.
    if models.get("fitness") is None:
        logger.error("=" * 70)
        logger.error("FITNESS PREDICTOR MODEL IS MISSING.")
        logger.error("  expected at: %s", config.FITNESS_PREDICTOR_PATH)
        logger.error("  this is the model used by step 3 of the hybrid GA flow.")
        logger.error("  the Go client's HealthCheck() will refuse to enable ANN.")
        logger.error("  fix: train it (notebooks/03_train_fitness_predictor.ipynb)")
        logger.error("       or copy an existing .keras file to that path.")
        logger.error("=" * 70)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _load_models_and_scalers()
    yield


# ── app ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Scheduling ANN API",
    description="API for ANN models assisting the Genetic Algorithm scheduling system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class ScheduleData(BaseModel):
    week_schedule: List[List[List[int]]]  # 6 days x 24 slots x 3 attributes
    curriculum_info: Optional[Dict[str, Any]] = None
    resources: Optional[Dict[str, Any]] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_schedule_keys(cls, data: Any):
        if not isinstance(data, dict):
            return data

        # Preferred key
        if data.get("week_schedule") is not None:
            return data

        # Alternate keys used by other sections/clients
        for key in ("section_schedule", "weekSchedule", "sectionSchedule"):
            if data.get(key) is not None:
                data = dict(data)
                data["week_schedule"] = data[key]
                return data

        # Nested payloads like {"schedule": {"week_schedule": [...]}}
        schedule_val = data.get("schedule")
        if schedule_val is None:
            return data
        if isinstance(schedule_val, list):
            data = dict(data)
            data["week_schedule"] = schedule_val
            return data
        if isinstance(schedule_val, dict):
            for key in ("week_schedule", "section_schedule", "weekSchedule", "sectionSchedule", "schedule"):
                if schedule_val.get(key) is not None:
                    data = dict(data)
                    data["week_schedule"] = schedule_val[key]
                    return data

        return data


class FitnessPredictionRequest(BaseModel):
    schedule: ScheduleData


class FitnessPredictionResponse(BaseModel):
    predicted_fitness: float
    confidence: Optional[float] = None
    processing_time_ms: float


class PreExtractedFeatures(BaseModel):
    features: List[float]


class FeatureBatchRequest(BaseModel):
    feature_vectors: List[PreExtractedFeatures]


class FitnessBatchPredictionResponse(BaseModel):
    predictions: List[FitnessPredictionResponse]
    processing_time_ms: float


class ConstraintCheckRequest(BaseModel):
    schedule: ScheduleData


class ConstraintCheckResponse(BaseModel):
    violations: Dict[str, bool]
    violation_scores: Dict[str, float]
    processing_time_ms: float


class ConstraintPredictionRequest(BaseModel):
    schedule: ScheduleData


class ConstraintPredictionResponse(BaseModel):
    instructor_conflict: float
    room_conflict: float
    no_lunch_break: float
    late_classes: float
    excessive_hours: float
    saturday_overload: float
    processing_time_ms: float


class ConstraintBatchRequest(BaseModel):
    schedules: List[ScheduleData]


class ConstraintBatchResponse(BaseModel):
    predictions: List[ConstraintPredictionResponse]
    processing_time_ms: float


class CrossoverRecommendationRequest(BaseModel):
    parent1: ScheduleData
    parent2: ScheduleData
    parent1_fitness: float
    parent2_fitness: float


class CrossoverRecommendationResponse(BaseModel):
    """Output of the crossover compatibility classifier.

    `compatible` is the binary decision (True iff `probability >= 0.5`);
    `probability` is the raw sigmoid output.  The model predicts whether
    the GA should *attempt* crossover between these parents or skip them,
    so callers should treat low probabilities as a signal to reroll the
    parent selection instead of wasting a crossover attempt.
    """

    compatible: bool
    probability: float
    processing_time_ms: float


class CrossoverCompatibilityRequest(BaseModel):
    parent1: ScheduleData
    parent2: ScheduleData


class CrossoverCompatibilityResponse(BaseModel):
    compatible: bool
    confidence: float
    processing_time_ms: float


class CrossoverBatchRequest(BaseModel):
    pairs: List[CrossoverCompatibilityRequest]


class CrossoverBatchResponse(BaseModel):
    predictions: List[CrossoverCompatibilityResponse]
    processing_time_ms: float


class MutationPredictionRequest(BaseModel):
    before_schedule: ScheduleData
    after_schedule: ScheduleData
    mutation_type: str
    before_fitness: float = 0.0
    after_fitness: float = 0.0


class MutationPredictionResponse(BaseModel):
    label: str = ""
    improve_prob: float = 0.0
    neutral_prob: float = 0.0
    worsen_prob: float = 0.0
    processing_time_ms: float = 0.0
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None


class MutationBatchRequest(BaseModel):
    predictions: List[MutationPredictionRequest]


class MutationBatchResponse(BaseModel):
    predictions: List[MutationPredictionResponse]
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    timestamp: str


# ── helper ────────────────────────────────────────────────────────────────────

def _elapsed_ms(start: datetime) -> float:
    return (datetime.now() - start).total_seconds() * 1000


_MUTATION_LABELS = ["improve", "neutral", "worsen"]
_CONSTRAINT_OUTPUT_NAMES = [
    "instructor_conflict",
    "room_conflict",
    "no_lunch_break",
    "late_classes",
    "excessive_hours",
    "saturday_overload",
]


def _run_constraint_batch(schedule_dicts: List[Dict]) -> List[Dict]:
    if models.get("constraint") is None:
        raise HTTPException(status_code=503, detail="Constraint classifier model not loaded")
    if feature_extractors.get("constraint") is None:
        raise HTTPException(status_code=503, detail="Constraint feature extractor not loaded")

    feature_list = [
        feature_extractors["constraint"].extract(schedule_dict)
        for schedule_dict in schedule_dicts
    ]
    if not feature_list:
        return []

    features = np.vstack(feature_list)
    if scalers.get("constraint") is not None:
        features = scalers["constraint"].transform(features)

    raw_predictions = models["constraint"].predict(features, verbose=0)
    rows = np.asarray(raw_predictions).reshape(len(schedule_dicts), -1)

    results: List[Dict[str, float]] = []
    for row in rows:
        result = {}
        for idx, name in enumerate(_CONSTRAINT_OUTPUT_NAMES):
            result[name] = float(row[idx]) if idx < row.size else 0.0
        results.append(result)
    return results


def _run_crossover_batch(pairs: List[Tuple[Dict, Dict]]) -> List[Dict]:
    if models.get("crossover") is None:
        raise HTTPException(status_code=503, detail="Crossover recommender model not loaded")
    if feature_extractors.get("crossover") is None:
        raise HTTPException(status_code=503, detail="Crossover feature extractor not loaded")

    feature_list = [
        feature_extractors["crossover"].extract(parent1, parent2, 0.0, 0.0)
        for parent1, parent2 in pairs
    ]
    if not feature_list:
        return []

    features = np.vstack(feature_list)
    if scalers.get("crossover") is not None:
        features = scalers["crossover"].transform(features)

    probabilities = models["crossover"].predict(features, verbose=0).reshape(-1)
    return [
        {
            "compatible": bool(prob >= 0.5),
            "confidence": float(prob),
        }
        for prob in probabilities
    ]


def _mutation_response_from_probs(probs: np.ndarray, processing_time_ms: float) -> MutationPredictionResponse:
    row = np.asarray(probs).reshape(-1)
    if row.size != len(_MUTATION_LABELS):
        raise ValueError(
            f"Mutation predictor returned {row.size} outputs; expected {len(_MUTATION_LABELS)}"
        )

    idx = int(np.argmax(row))
    label = _MUTATION_LABELS[idx]
    probability_map = {
        label_name: float(prob) for label_name, prob in zip(_MUTATION_LABELS, row)
    }
    return MutationPredictionResponse(
        label=label,
        improve_prob=probability_map["improve"],
        neutral_prob=probability_map["neutral"],
        worsen_prob=probability_map["worsen"],
        processing_time_ms=processing_time_ms,
        prediction=label,
        confidence=float(row[idx]),
        probabilities=probability_map,
    )


def _run_mutation_batch(mutation_requests: List[Dict]) -> List[np.ndarray]:
    if models.get("mutation") is None:
        raise HTTPException(status_code=503, detail="Mutation predictor model not loaded")
    if scalers.get("mutation") is None:
        raise HTTPException(status_code=503, detail="Mutation scaler not loaded")
    if feature_extractors.get("mutation") is None:
        raise HTTPException(status_code=503, detail="Mutation feature extractor not loaded")

    feature_list = [
        feature_extractors["mutation"].extract(
            request["before_schedule"],
            request["after_schedule"],
            request.get("mutation_type", ""),
            request.get("before_fitness", 0.0),
            request.get("after_fitness", 0.0),
        )
        for request in mutation_requests
    ]
    if not feature_list:
        return []

    features = np.vstack(feature_list)
    features = scalers["mutation"].transform(features)
    return list(models["mutation"].predict(features, verbose=0))


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "Scheduling ANN API",
        "version": "1.0.0",
        "endpoints": {
            "health":                         "/health",
            "fitness":                        "/predict/fitness",
            "fitness_batch":                  "/predict/fitness/batch",
            "fitness_batch_preextracted":     "/predict/fitness/batch/preextracted",
            "constraints":                    "/predict/constraints",
            "constraint_batch":               "/predict/constraint/batch",
            "crossover":                      "/recommend/crossover",
            "crossover_batch":                "/predict/crossover/batch",
            "mutation":                       "/predict/mutation",
            "mutation_batch":                 "/predict/mutation/batch",
        },
    }


# FIX 1: status now reflects actual model availability.
#   "healthy"   — all 4 models loaded
#   "degraded"  — at least 1 model loaded
#   "unhealthy" — no models loaded
@app.get("/health", response_model=HealthResponse)
async def health_check():
    loaded_map = {
        "fitness_predictor":    models.get("fitness")    is not None,
        "constraint_classifier":models.get("constraint") is not None,
        "crossover_recommender":models.get("crossover")  is not None,
        "mutation_predictor":   models.get("mutation")   is not None,
        "feature_scaler":       scalers.get("features")  is not None,
    }
    n_loaded = sum(loaded_map.values())

    if n_loaded == len(loaded_map):
        status = "healthy"
    elif n_loaded > 0:
        status = "degraded"
    else:
        status = "unhealthy"

    if status != "healthy":
        missing = [name for name, ok in loaded_map.items() if not ok]
        logger.warning(f"health_check: status={status!r}, missing models: {missing}")

    return HealthResponse(
        status=status,
        models_loaded=loaded_map,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict/fitness", response_model=FitnessPredictionResponse)
async def predict_fitness(request: FitnessPredictionRequest):
    start_time = datetime.now()
    ws = request.schedule.week_schedule
    logger.info(f"/predict/fitness — schedule shape: {len(ws)}d×{len(ws[0]) if ws else '?'}s")

    if models.get("fitness") is None:
        raise HTTPException(status_code=503, detail="Fitness predictor model not loaded")
    if scalers.get("features") is None or scalers.get("fitness") is None:
        raise HTTPException(status_code=503, detail="Feature or fitness scaler not loaded")

    try:
        schedule_dict = request.schedule.model_dump()
        features = feature_extractors["fitness"].extract(schedule_dict)
        features_normalized = scalers["features"].transform(features.reshape(1, -1))
        prediction_normalized = models["fitness"].predict(features_normalized, verbose=0)
        prediction = scalers["fitness"].inverse_transform(prediction_normalized)[0][0]
        ms = _elapsed_ms(start_time)
        logger.info(f"/predict/fitness — predicted={prediction:.4f} ({ms:.1f} ms)")
        return FitnessPredictionResponse(
            predicted_fitness=float(prediction),
            confidence=0.0,
            processing_time_ms=ms,
        )
    except Exception as exc:
        logger.error(f"/predict/fitness — error: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.post("/predict/fitness/batch", response_model=List[FitnessPredictionResponse])
async def predict_fitness_batch(requests: Any = Body(...)):
    """Predict fitness for a batch of schedules.

    Accepted shapes:
      {"schedules": [{"schedule": {"week_schedule": [...]}}, ...]}   ← Go client
      {"requests":  [{"schedule": {"week_schedule": [...]}}, ...]}
      [{"schedule": {"week_schedule": [...]}}, ...]
      {"schedule": {"week_schedule": [...]}}  (single item)
    """
    start_time = datetime.now()

    if models.get("fitness") is None:
        raise HTTPException(status_code=503, detail="Fitness predictor model not loaded")
    if scalers.get("features") is None or scalers.get("fitness") is None:
        raise HTTPException(status_code=503, detail="Feature or fitness scaler not loaded")

    try:
        requests_parsed: List[FitnessPredictionRequest] = []

        if isinstance(requests, dict) and "schedules" in requests:
            for item in (requests.get("schedules") or []):
                if isinstance(item, dict) and "schedule" in item:
                    requests_parsed.append(FitnessPredictionRequest.model_validate(item))
                else:
                    requests_parsed.append(FitnessPredictionRequest.model_validate({"schedule": item}))
        elif isinstance(requests, dict) and "requests" in requests:
            for item in (requests.get("requests") or []):
                if isinstance(item, dict) and "schedule" in item:
                    requests_parsed.append(FitnessPredictionRequest.model_validate(item))
                else:
                    requests_parsed.append(FitnessPredictionRequest.model_validate({"schedule": item}))
        elif isinstance(requests, dict):
            requests_parsed = [FitnessPredictionRequest.model_validate(requests)]
        else:
            for r in requests:
                if isinstance(r, FitnessPredictionRequest):
                    requests_parsed.append(r)
                else:
                    requests_parsed.append(FitnessPredictionRequest.model_validate(r))

        logger.info(f"/predict/fitness/batch — {len(requests_parsed)} schedules received")

        # FIX 4: guard against empty batch so np.vstack([]) is never called.
        if not requests_parsed:
            logger.info("/predict/fitness/batch — empty batch, returning []")
            return []

        feature_list = []
        for req in requests_parsed:
            feat = feature_extractors["fitness"].extract(req.schedule.model_dump())
            feature_list.append(feat)

        features_array = np.vstack(feature_list)
        features_normalized = scalers["features"].transform(features_array)
        predictions_normalized = models["fitness"].predict(features_normalized, verbose=0)
        predictions = scalers["fitness"].inverse_transform(predictions_normalized).reshape(-1)

        # FIX 3: compute total processing time once outside the loop.
        ms = _elapsed_ms(start_time)
        logger.info(
            f"/predict/fitness/batch — {len(predictions)} predictions, "
            f"min={float(predictions.min()):.4f} max={float(predictions.max()):.4f} ({ms:.1f} ms)"
        )
        return [
            FitnessPredictionResponse(
                predicted_fitness=float(pred),
                confidence=0.0,
                processing_time_ms=ms,
            )
            for pred in predictions
        ]

    except Exception as exc:
        logger.error(f"/predict/fitness/batch — error: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}")


@app.post("/predict/fitness/batch/preextracted", response_model=FitnessBatchPredictionResponse)
async def predict_fitness_batch_preextracted(request: FeatureBatchRequest):
    start_time = datetime.now()

    if models.get("fitness") is None:
        raise HTTPException(status_code=503, detail="Fitness predictor model not loaded")
    if scalers.get("features") is None or scalers.get("fitness") is None:
        raise HTTPException(status_code=503, detail="Feature or fitness scaler not loaded")

    try:
        if not request.feature_vectors:
            return FitnessBatchPredictionResponse(predictions=[], processing_time_ms=0.0)

        features = np.asarray(
            [item.features for item in request.feature_vectors],
            dtype=np.float32,
        )
        features_normalized = scalers["features"].transform(features)
        predictions_normalized = models["fitness"].predict(features_normalized, verbose=0)
        predictions = scalers["fitness"].inverse_transform(predictions_normalized).reshape(-1)
        ms = _elapsed_ms(start_time)
        return FitnessBatchPredictionResponse(
            predictions=[
                FitnessPredictionResponse(
                    predicted_fitness=float(pred),
                    confidence=0.0,
                    processing_time_ms=ms,
                )
                for pred in predictions
            ],
            processing_time_ms=ms,
        )
    except Exception as exc:
        logger.error(f"/predict/fitness/batch/preextracted failed: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Pre-extracted batch prediction failed: {exc}")


@app.post("/predict/constraints", response_model=ConstraintCheckResponse)
async def check_constraints(request: ConstraintCheckRequest):
    start_time = datetime.now()
    ws = request.schedule.week_schedule
    logger.info(f"/predict/constraints — schedule shape: {len(ws)}d×{len(ws[0]) if ws else '?'}s")

    if models.get("constraint") is None:
        raise HTTPException(status_code=503, detail="Constraint classifier model not loaded")

    try:
        schedule_dict = request.schedule.model_dump()
        features = feature_extractors["constraint"].extract(schedule_dict)

        if scalers.get("constraint") is not None:
            features = scalers["constraint"].transform(features.reshape(1, -1))
        else:
            features = features.reshape(1, -1)

        predictions = models["constraint"].predict(features, verbose=0)[0]

        constraint_names = [
            "instructor_conflict", "room_conflict", "no_lunch_break",
            "late_classes", "excessive_hours", "saturday_overload",
            "resource_unavailable", "curriculum_conflict",
            "room_capacity", "instructor_availability",
        ]
        violations       = {n: bool(p > 0.5)  for n, p in zip(constraint_names, predictions)}
        violation_scores = {n: float(p)        for n, p in zip(constraint_names, predictions)}
        n_violated = sum(violations.values())
        ms = _elapsed_ms(start_time)
        logger.info(
            f"/predict/constraints — {n_violated}/{len(constraint_names)} violated "
            f"({[n for n,v in violations.items() if v] or 'none'}) ({ms:.1f} ms)"
        )
        return ConstraintCheckResponse(
            violations=violations,
            violation_scores=violation_scores,
            processing_time_ms=ms,
        )
    except Exception as exc:
        logger.error(f"/predict/constraints — error: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Constraint checking failed: {exc}")


@app.post("/predict/constraint/batch", response_model=ConstraintBatchResponse)
async def predict_constraint_batch(request: ConstraintBatchRequest):
    start_time = datetime.now()

    try:
        schedule_dicts = [schedule.model_dump() for schedule in request.schedules]
        raw_predictions = _run_constraint_batch(schedule_dicts)
        ms = _elapsed_ms(start_time)
        return ConstraintBatchResponse(
            predictions=[
                ConstraintPredictionResponse(
                    instructor_conflict=pred["instructor_conflict"],
                    room_conflict=pred["room_conflict"],
                    no_lunch_break=pred["no_lunch_break"],
                    late_classes=pred["late_classes"],
                    excessive_hours=pred["excessive_hours"],
                    saturday_overload=pred["saturday_overload"],
                    processing_time_ms=ms,
                )
                for pred in raw_predictions
            ],
            processing_time_ms=ms,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"/predict/constraint/batch failed: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Constraint batch prediction failed: {exc}")


@app.post("/recommend/crossover", response_model=CrossoverRecommendationResponse)
async def recommend_crossover(request: CrossoverRecommendationRequest):
    """Binary crossover-compatibility classifier.

    Returns the predicted probability that crossing the two parent schedules
    will produce a valid offspring with fitness above the training threshold.
    GA callers should treat low probabilities as a hint to pick different
    parents instead of attempting (and likely retrying) the crossover.
    """
    start_time = datetime.now()
    logger.info(
        f"/recommend/crossover — parent1_fitness={request.parent1_fitness:.4f} "
        f"parent2_fitness={request.parent2_fitness:.4f}"
    )

    if models.get("crossover") is None:
        raise HTTPException(status_code=503, detail="Crossover recommender model not loaded")
    if feature_extractors.get("crossover") is None:
        raise HTTPException(status_code=503, detail="Crossover feature extractor not loaded")

    try:
        features = feature_extractors["crossover"].extract(
            request.parent1.model_dump(),
            request.parent2.model_dump(),
            request.parent1_fitness,
            request.parent2_fitness,
        ).reshape(1, -1)

        if scalers.get("crossover") is not None:
            features = scalers["crossover"].transform(features)

        probability = float(models["crossover"].predict(features, verbose=0).reshape(-1)[0])
        compatible = bool(probability >= 0.5)
        ms = _elapsed_ms(start_time)

        logger.info(
            f"/recommend/crossover — compatible={compatible} "
            f"p={probability:.4f} ({ms:.1f} ms)"
        )
        return CrossoverRecommendationResponse(
            compatible=compatible,
            probability=probability,
            processing_time_ms=ms,
        )
    except Exception as exc:
        logger.error(f"/recommend/crossover — error: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Crossover recommendation failed: {exc}")


@app.post("/predict/crossover/batch", response_model=CrossoverBatchResponse)
async def predict_crossover_batch(request: CrossoverBatchRequest):
    start_time = datetime.now()

    try:
        pairs = [
            (pair.parent1.model_dump(), pair.parent2.model_dump())
            for pair in request.pairs
        ]
        raw_predictions = _run_crossover_batch(pairs)
        ms = _elapsed_ms(start_time)
        return CrossoverBatchResponse(
            predictions=[
                CrossoverCompatibilityResponse(
                    compatible=pred["compatible"],
                    confidence=pred["confidence"],
                    processing_time_ms=ms,
                )
                for pred in raw_predictions
            ],
            processing_time_ms=ms,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"/predict/crossover/batch failed: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Crossover batch prediction failed: {exc}")


@app.post("/predict/mutation", response_model=MutationPredictionResponse)
async def predict_mutation(request: MutationPredictionRequest):
    start_time = datetime.now()

    if models.get("mutation") is None:
        raise HTTPException(status_code=503, detail="Mutation predictor model not loaded")
    if scalers.get("mutation") is None:
        raise HTTPException(status_code=503, detail="Mutation scaler not loaded")
    if feature_extractors.get("mutation") is None:
        raise HTTPException(status_code=503, detail="Mutation feature extractor not loaded")

    try:
        features = feature_extractors["mutation"].extract(
            request.before_schedule.model_dump(),
            request.after_schedule.model_dump(),
            request.mutation_type,
            request.before_fitness,
            request.after_fitness,
        ).reshape(1, -1)

        features = scalers["mutation"].transform(features)
        probs = models["mutation"].predict(features, verbose=0).reshape(-1)
        if probs.size != len(_MUTATION_LABELS):
            raise ValueError(
                f"Mutation predictor returned {probs.size} outputs; expected {len(_MUTATION_LABELS)}"
            )

        idx = int(np.argmax(probs))
        prediction = _MUTATION_LABELS[idx]
        confidence = float(probs[idx])
        probability_map = {
            label: float(prob) for label, prob in zip(_MUTATION_LABELS, probs)
        }
        ms = _elapsed_ms(start_time)
        logger.info(
            f"/predict/mutation — prediction={prediction} conf={confidence:.4f} ({ms:.1f} ms)"
        )
        return MutationPredictionResponse(
            label=prediction,
            improve_prob=probability_map.get("improve", 0.0),
            neutral_prob=probability_map.get("neutral", 0.0),
            worsen_prob=probability_map.get("worsen", 0.0),
            prediction=prediction,
            confidence=confidence,
            probabilities=probability_map,
            processing_time_ms=ms,
        )
    except Exception as exc:
        logger.error(f"/predict/mutation — error: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Mutation prediction failed: {exc}")


# ── entrypoint ────────────────────────────────────────────────────────────────

@app.post("/predict/mutation/batch", response_model=MutationBatchResponse)
async def predict_mutation_batch(request: MutationBatchRequest):
    start_time = datetime.now()

    try:
        mutation_dicts = [item.model_dump() for item in request.predictions]
        raw_predictions = _run_mutation_batch(mutation_dicts)
        ms = _elapsed_ms(start_time)
        return MutationBatchResponse(
            predictions=[
                _mutation_response_from_probs(probs, ms)
                for probs in raw_predictions
            ],
            processing_time_ms=ms,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"/predict/mutation/batch failed: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Mutation batch prediction failed: {exc}")


if __name__ == "__main__":
    import uvicorn

    print("=" * 70)
    print("Starting Scheduling ANN API Server")
    print("=" * 70)
    print(f"Host: {config.API_HOST}")
    print(f"Port: {config.API_PORT}")
    print(f"Workers: {config.API_WORKERS}")
    print("=" * 70)

    def _check_port_available(host: str, port: int) -> bool:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((host, port))
            s.close()
            return True
        except OSError:
            return False

    if not _check_port_available(config.API_HOST, config.API_PORT):
        print(f"ERROR: Port {config.API_PORT} on {config.API_HOST} is already in use.")
        print("Possible causes: another API instance is running or the port is reserved.")
        print("To diagnose: run `netstat -ano | findstr :<port>` and `tasklist /FI \"PID eq <pid>\"` on Windows.")
        print("Or start the API on a different port by setting the ANN_API_PORT environment variable.")
        sys.exit(1)

    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        workers=config.API_WORKERS,
        reload=False,
    )
