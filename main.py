"""
Hospital AI Backend - FastAPI Application

Provides endpoints for medical image classification and clinical risk
assessment.
"""

import io
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional, List
from datetime import datetime

# Suppress TensorFlow warnings before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=ALL, 1=INFO, 2=WARNING, 3=ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom ops

import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from config import config

# Load environment variables
load_dotenv()

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


class CombinedPredictionResponse(BaseModel):
    """Combined risk assessment response"""
    overall_risk_score: float
    risk_level: str
    image_prediction: str
    image_confidence: float
    clinical_prediction: str
    clinical_confidence: float
    recommendations: List[str]


class ChatMessage(BaseModel):
    """Chat message for doctor AI agent"""
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response from doctor AI agent"""
    response: str
    session_id: str
    timestamp: str


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


# ---------------- AI DOCTOR AGENT ----------------
# Store conversation sessions
conversation_sessions = {}  # Stores {session_id: {"llm": ChatOpenAI, "history": [messages]}}

SYSTEM_PROMPT = """You are a compassionate and knowledgeable AI medical assistant. 
Your role is to:
- Provide clear, evidence-based medical information
- Help patients understand their risk assessments and medical test results
- Offer general health guidance and lifestyle recommendations
- Answer questions about symptoms, conditions, and preventive care
- Emphasize when professional medical consultation is necessary

Important reminders:
- You provide educational information, not definitive diagnoses
- Always recommend consulting healthcare professionals for serious concerns
- Be empathetic and supportive in your responses
- Focus on evidence-based medical information

Current context: You're assisting with a patient who has received risk assessments based on medical imaging and clinical data."""


def get_ai_doctor_agent():
    """Initialize LangChain AI doctor agent"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
            return None
        
        llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-4",
            openai_api_key=api_key
        )
        
        return {"llm": llm, "history": [SystemMessage(content=SYSTEM_PROMPT)]}
        
    except Exception as e:
        logger.error(f"Failed to initialize AI doctor agent: {e}")
        return None


def calculate_combined_risk_score(
    image_pred: str,
    image_conf: float,
    clinical_pred: str,
    clinical_conf: float
) -> tuple[float, str, List[str]]:
    """
    Calculate overall risk score from image and clinical predictions
    
    Returns: (risk_score, risk_level, recommendations)
    """
    # Convert predictions to risk scores (0-100)
    image_risk = 0
    if image_pred in ["breast_cancer", "lung_cancer"]:
        image_risk = image_conf * 100
    else:  # normal
        image_risk = (1 - image_conf) * 100
    
    clinical_risk = 0
    if "High Risk" in clinical_pred:
        clinical_risk = clinical_conf * 100
    else:
        clinical_risk = (1 - clinical_conf) * 100
    
    # Weighted average (60% image, 40% clinical)
    overall_risk = (image_risk * 0.6) + (clinical_risk * 0.4)
    
    # Determine risk level
    if overall_risk < 30:
        risk_level = "Low Risk"
    elif overall_risk < 60:
        risk_level = "Moderate Risk"
    else:
        risk_level = "High Risk"
    
    # Generate recommendations
    recommendations = []
    
    if risk_level == "High Risk":
        recommendations.append("⚠️ Immediate consultation with a healthcare provider is recommended")
        recommendations.append("Consider scheduling a comprehensive medical examination")
        recommendations.append("Bring all test results and imaging to your appointment")
    elif risk_level == "Moderate Risk":
        recommendations.append("Schedule a follow-up appointment with your doctor")
        recommendations.append("Monitor your symptoms and any changes")
        recommendations.append("Consider lifestyle modifications for prevention")
    else:
        recommendations.append("Continue regular health check-ups")
        recommendations.append("Maintain a healthy lifestyle")
        recommendations.append("Stay informed about preventive care")
    
    # Add specific recommendations based on findings
    if image_pred != "normal":
        recommendations.append(f"Discuss {image_pred.replace('_', ' ')} findings with a specialist")
    
    if "High Risk" in clinical_pred:
        recommendations.append("Review your clinical parameters with your physician")
        recommendations.append("Consider additional diagnostic tests as recommended")
    
    return round(overall_risk, 2), risk_level, recommendations


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


@app.post("/predict-combined", response_model=CombinedPredictionResponse)
async def predict_combined(
    file: UploadFile = File(...),
    age: int = Form(...),
    gender: int = Form(...),
    smoking: int = Form(...),
    alcohol: int = Form(...),
    bmi: float = Form(...),
    wbc: float = Form(...),
    hemoglobin: float = Form(...),
    platelets: int = Form(...)
):
    """
    Combined prediction using both medical image and clinical data.
    Returns comprehensive risk assessment with recommendations.
    """
    if image_model is None or clinical_model is None:
        raise HTTPException(
            status_code=503,
            detail="Models not available"
        )
    
    try:
        # 1. Process image
        if not file.content_type or file.content_type not in config.api.ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file type"
            )
        
        contents = await file.read()
        if len(contents) > config.api.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File too large"
            )
        
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize(config.model.IMAGE_SIZE)
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Image prediction
        image_predictions = image_model.predict(
            image_array,
            verbose=config.performance.TF_PREDICT_VERBOSE
        )
        image_idx = int(np.argmax(image_predictions[0]))
        image_confidence = float(image_predictions[0][image_idx])
        image_result = config.model.CLASS_NAMES[image_idx]
        
        # 2. Process clinical data
        clinical_data = ClinicalInput(
            age=age,
            gender=gender,
            smoking=smoking,
            alcohol=alcohol,
            bmi=bmi,
            wbc=wbc,
            hemoglobin=hemoglobin,
            platelets=platelets
        )
        
        features = np.array([[
            clinical_data.age,
            clinical_data.gender,
            clinical_data.smoking,
            clinical_data.alcohol,
            clinical_data.bmi,
            clinical_data.wbc,
            clinical_data.hemoglobin,
            clinical_data.platelets
        ]], dtype=np.float32)
        
        # Clinical prediction
        clinical_prediction = clinical_model.predict(features)[0]
        clinical_probabilities = clinical_model.predict_proba(features)[0]
        clinical_confidence = float(clinical_probabilities[clinical_prediction])
        clinical_result = config.model.RISK_HIGH if clinical_prediction == 1 else config.model.RISK_LOW
        
        # 3. Calculate combined risk
        risk_score, risk_level, recommendations = calculate_combined_risk_score(
            image_result,
            image_confidence,
            clinical_result,
            clinical_confidence
        )
        
        logger.info(
            f"Combined prediction - Risk Score: {risk_score}, "
            f"Level: {risk_level}, Image: {image_result}, Clinical: {clinical_result}"
        )
        
        return {
            "overall_risk_score": risk_score,
            "risk_level": risk_level,
            "image_prediction": image_result,
            "image_confidence": round(image_confidence, 3),
            "clinical_prediction": clinical_result,
            "clinical_confidence": round(clinical_confidence, 3),
            "recommendations": recommendations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Combined prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/chat", response_model=ChatResponse)
async def chat_with_doctor(chat_input: ChatMessage):
    """
    Chat with AI doctor agent for medical advice and consultation.
    """
    try:
        session_id = chat_input.session_id or f"session_{datetime.now().timestamp()}"
        
        # Get or create conversation for this session
        if session_id not in conversation_sessions:
            agent = get_ai_doctor_agent()
            if agent is None:
                raise HTTPException(
                    status_code=503,
                    detail="AI doctor agent not available. Please set OPENAI_API_KEY."
                )
            conversation_sessions[session_id] = agent
        
        session = conversation_sessions[session_id]
        llm = session["llm"]
        history = session["history"]
        
        # Add user message to history
        history.append(HumanMessage(content=chat_input.message))
        
        # Get response from AI agent
        ai_response = llm.invoke(history)
        response = ai_response.content
        
        # Add AI response to history
        history.append(AIMessage(content=response))
        
        logger.info(f"Chat session {session_id}: User message length: {len(chat_input.message)}")
        
        return {
            "response": response,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )


@app.delete("/chat/{session_id}")
async def end_chat_session(session_id: str):
    """
    End a chat session and clear conversation history.
    """
    if session_id in conversation_sessions:
        del conversation_sessions[session_id]
        logger.info(f"Chat session {session_id} ended")
        return {"status": "success", "message": "Session ended"}
    else:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
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
