"""
Configuration Management for Hospital AI Backend
Centralizes all constants and settings for easy management and deployment.
"""

import os
from pathlib import Path

# ---------------- PROJECT PATHS ----------------
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
TRAINING_DIR = BASE_DIR / "training"


# ---------------- MODEL SETTINGS ----------------


class ModelConfig:
    """Model-related configuration"""
    
    # Paths
    IMAGE_MODEL_PATH = MODEL_DIR / "image_model.h5"
    CLINICAL_MODEL_PATH = MODEL_DIR / "clinical_model.pkl"
    
    # Image model settings
    IMAGE_SIZE = (224, 224)
    IMAGE_CHANNELS = 3
    
    # Class names for image classification
    CLASS_NAMES = ["breast_cancer", "lung_cancer", "normal"]
    
    # Clinical features (in order)
    CLINICAL_FEATURES = [
        "age", "gender", "smoking", "alcohol",
        "bmi", "wbc", "hemoglobin", "platelets"
    ]
    
    # Risk categories
    RISK_HIGH = "High Risk"
    RISK_LOW = "Low Risk"


# ---------------- TRAINING SETTINGS ----------------
class TrainingConfig:
    """Training-related configuration"""
    
    # Image training
    IMG_BATCH_SIZE = 16
    IMG_EPOCHS = 10
    IMG_VALIDATION_SPLIT = 0.2
    IMG_DATASET_PATH = TRAINING_DIR / "image_training" / "dataset"
    
    # Clinical training
    CLINICAL_TEST_SIZE = 0.2
    CLINICAL_RANDOM_STATE = 42
    RANDOM_FOREST_ESTIMATORS = 100
    
    # Data augmentation
    USE_AUGMENTATION = True
    ROTATION_RANGE = 20
    ZOOM_RANGE = 0.2
    HORIZONTAL_FLIP = True


# ---------------- API SETTINGS ----------------
class APIConfig:
    """API server configuration"""
    
    # Server settings
    HOST = os.getenv("API_HOST", "0.0.0.0")
    PORT = int(os.getenv("API_PORT", "8000"))
    RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"
    
    # CORS settings
    CORS_ORIGINS = os.getenv(
        "CORS_ORIGINS",
        "*"  # In production, specify allowed origins
    ).split(",")
    CORS_ALLOW_CREDENTIALS = True
    CORS_ALLOW_METHODS = ["*"]
    CORS_ALLOW_HEADERS = ["*"]
    
    # API metadata
    TITLE = "Hospital AI Backend"
    DESCRIPTION = "AI-powered medical diagnosis API"
    VERSION = "1.0.0"
    
    # File upload limits
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_IMAGE_TYPES = [
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/bmp",
        "image/webp"
    ]


# ---------------- LOGGING SETTINGS ----------------
class LoggingConfig:
    """Logging configuration"""
    
    LEVEL = os.getenv("LOG_LEVEL", "INFO")
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # File logging (optional)
    LOG_TO_FILE = os.getenv("LOG_TO_FILE", "false").lower() == "true"
    LOG_FILE_PATH = BASE_DIR / "logs" / "app.log"
    LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10MB
    LOG_FILE_BACKUP_COUNT = 5


# ---------------- VALIDATION SETTINGS ----------------
class ValidationConfig:
    """Input validation constraints"""
    
    # Clinical data constraints
    AGE_MIN = 0
    AGE_MAX = 120
    
    GENDER_VALUES = [0, 1]  # 0=Female, 1=Male
    BINARY_VALUES = [0, 1]  # For yes/no fields
    
    BMI_MIN = 10.0
    BMI_MAX = 100.0
    
    WBC_MIN = 0.0
    WBC_MAX = 50.0  # Normal range is 4-11, but allowing higher for abnormal cases
    
    HEMOGLOBIN_MIN = 0.0
    HEMOGLOBIN_MAX = 25.0  # Normal range varies by gender
    
    PLATELETS_MIN = 0
    PLATELETS_MAX = 1000000


# ---------------- PERFORMANCE SETTINGS ----------------
class PerformanceConfig:
    """Performance and optimization settings"""
    
    # Model inference
    TF_PREDICT_VERBOSE = 0  # Silent predictions
    TF_NUM_THREADS = os.getenv("TF_NUM_THREADS", None)
    
    # Caching
    ENABLE_MODEL_CACHE = True
    CACHE_PREDICTIONS = False  # Disable for medical applications
    
    # Concurrency
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))


# ---------------- ENVIRONMENT ----------------
class Environment:
    """Environment detection"""
    
    ENV = os.getenv("ENVIRONMENT", "development")
    
    @classmethod
    def is_development(cls) -> bool:
        return cls.ENV == "development"
    
    @classmethod
    def is_production(cls) -> bool:
        return cls.ENV == "production"
    
    @classmethod
    def is_testing(cls) -> bool:
        return cls.ENV == "testing"


# ---------------- EXPORT MAIN CONFIG ----------------
class Config:
    """Main configuration class - aggregates all settings"""
    
    model = ModelConfig
    training = TrainingConfig
    api = APIConfig
    logging = LoggingConfig
    validation = ValidationConfig
    performance = PerformanceConfig
    environment = Environment
    
    # Quick access paths
    BASE_DIR = BASE_DIR
    MODEL_DIR = MODEL_DIR


# Export for easy import
config = Config()
