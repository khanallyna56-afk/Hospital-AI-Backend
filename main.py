"""
Hospital AI Backend - FastAPI Application

Provides endpoints for medical image classification and clinical risk
assessment.
"""

import io
import logging
import os
from logging.handlers import RotatingFileHandler

# Suppress TensorFlow warnings before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=ALL, 1=INFO, 2=WARNING, 3=ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom ops

import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from config import config

# Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')

# ---------------- LOGGING ----------------
# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, config.logging.LEVEL),
    format=config.logging.FORMAT,
    datefmt=config.logging.DATE_FORMAT
)

# Optional file logging
if config.logging.LOG_TO_FILE:
    config.logging.LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        config.logging.LOG_FILE_PATH,
        maxBytes=config.logging.LOG_FILE_MAX_BYTES,
        backupCount=config.logging.LOG_FILE_BACKUP_COUNT
    )
    file_handler.setFormatter(
        logging.Formatter(config.logging.FORMAT, config.logging.DATE_FORMAT)
    )
    logger.addHandler(file_handler)

# ---------------- APP INITIALIZATION ----------------
app = FastAPI(
    title=config.api.TITLE,
    description=config.api.DESCRIPTION,
    version=config.api.VERSION
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.CORS_ORIGINS,
    allow_credentials=config.api.CORS_ALLOW_CREDENTIALS,
    allow_methods=config.api.CORS_ALLOW_METHODS,
    allow_headers=config.api.CORS_ALLOW_HEADERS,
)


# ---------------- DATA MODELS ----------------


class ClinicalInput(BaseModel):
    """Clinical data input for risk assessment"""
    age: int = Field(
        ...,
        ge=config.validation.AGE_MIN,
        le=config.validation.AGE_MAX,
        description="Patient age"
    )
    gender: int = Field(..., ge=0, le=1, description="1=Male, 0=Female")
    smoking: int = Field(..., ge=0, le=1, description="1=Yes, 0=No")
    alcohol: int = Field(..., ge=0, le=1, description="1=Yes, 0=No")
    bmi: float = Field(
        ...,
        ge=config.validation.BMI_MIN,
        le=config.validation.BMI_MAX,
        description="Body Mass Index"
    )
    wbc: float = Field(
        ...,
        ge=config.validation.WBC_MIN,
        le=config.validation.WBC_MAX,
        description="White Blood Cell count"
    )
    hemoglobin: float = Field(
        ...,
        ge=config.validation.HEMOGLOBIN_MIN,
        le=config.validation.HEMOGLOBIN_MAX,
        description="Hemoglobin level"
    )
    platelets: int = Field(
        ...,
        ge=config.validation.PLATELETS_MIN,
        le=config.validation.PLATELETS_MAX,
        description="Platelet count"
    )


class PredictionResponse(BaseModel):
    """Standard prediction response"""
    prediction: str
    confidence: float


# ---------------- MODEL LOADING ----------------
def load_models():
    """Load ML models at startup"""
    try:
        # Ensure model directory exists
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Load image classification model
        image_model_path = config.model.IMAGE_MODEL_PATH
        if not image_model_path.exists():
            raise FileNotFoundError(
                f"Image model not found: {image_model_path}. "
                f"Please train the model first using training scripts."
            )

        image_model = tf.keras.models.load_model(
            str(image_model_path),
            compile=False
        )
        image_model.compile(
            optimizer="adam",
            loss="categorical_crossentropy"
        )
        logger.info(f"✓ Image model loaded from {image_model_path}")

        # Load clinical risk model
        clinical_model_path = config.model.CLINICAL_MODEL_PATH
        if not clinical_model_path.exists():
            raise FileNotFoundError(
                f"Clinical model not found: {clinical_model_path}. "
                f"Please train the model first using training scripts."
            )

        clinical_model = joblib.load(str(clinical_model_path))
        logger.info(f"✓ Clinical model loaded from {clinical_model_path}")

        return image_model, clinical_model

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


# Global model references
try:
    image_model, clinical_model = load_models()
except Exception as e:
    logger.warning(
        f"Models not loaded: {e}. "
        "Endpoints will fail until models are available."
    )
    image_model, clinical_model = None, None


# ---------------- ROUTES ----------------
@app.get("/")
def root():
    """Health check endpoint"""
    models_loaded = image_model is not None and clinical_model is not None
    return {
        "status": "running",
        "service": config.api.TITLE,
        "version": config.api.VERSION,
        "environment": config.environment.ENV,
        "models_loaded": models_loaded
    }


@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "image_model": image_model is not None,
        "clinical_model": clinical_model is not None
    }


@app.post("/predict-image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """
    Predict cancer type from medical image.

    Args:
        file: Medical image file (JPEG, PNG, etc.)

    Returns:
        Prediction and confidence score
    """
    if image_model is None:
        raise HTTPException(
            status_code=503,
            detail="Image model not available"
        )

    try:
        # Validate file type
        if not file.content_type:
            raise HTTPException(
                status_code=400,
                detail="File type not specified"
            )

        if file.content_type not in config.api.ALLOWED_IMAGE_TYPES:
            allowed = ', '.join(config.api.ALLOWED_IMAGE_TYPES)
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {allowed}"
            )

        # Read and validate file size
        contents = await file.read()
        if len(contents) > config.api.MAX_FILE_SIZE:
            max_mb = config.api.MAX_FILE_SIZE / 1024 / 1024
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {max_mb}MB"
            )

        # Preprocess image
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize(config.model.IMAGE_SIZE)

        # Normalize and prepare for inference
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Predict
        predictions = image_model.predict(
            image_array,
            verbose=config.performance.TF_PREDICT_VERBOSE
        )
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_idx])

        result = config.model.CLASS_NAMES[predicted_idx]
        logger.info(
            f"Image prediction: {result} (confidence: {confidence:.3f})"
        )

        return {
            "prediction": result,
            "confidence": round(confidence, 3)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict-clinical", response_model=PredictionResponse)
def predict_clinical(data: ClinicalInput):
    """
    Predict health risk from clinical data.

    Args:
        data: Clinical parameters (age, BMI, blood work, etc.)

    Returns:
        Risk assessment and confidence
    """
    if clinical_model is None:
        raise HTTPException(
            status_code=503,
            detail="Clinical model not available"
        )

    try:
        # Prepare features
        features = np.array([[
            data.age,
            data.gender,
            data.smoking,
            data.alcohol,
            data.bmi,
            data.wbc,
            data.hemoglobin,
            data.platelets
        ]], dtype=np.float32)

        # Predict
        prediction = clinical_model.predict(features)[0]
        probabilities = clinical_model.predict_proba(features)[0]
        confidence = float(probabilities[prediction])

        if prediction == 1:
            result = config.model.RISK_HIGH
        else:
            result = config.model.RISK_LOW

        logger.info(
            f"Clinical prediction: {result} (confidence: {confidence:.2f})"
        )

        return {
            "prediction": result,
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        logger.error(f"Clinical prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ---------------- ENTRY POINT ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.api.HOST,
        port=config.api.PORT,
        reload=config.api.RELOAD,
        log_level=config.logging.LEVEL.lower()
    )
