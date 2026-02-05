# 🏥 Hospital AI Backend

AI-powered medical diagnosis API providing image-based cancer classification and clinical risk assessment using deep learning and machine learning models.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Training Models](#training-models)
- [Deployment](#deployment)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## ✨ Features

### 🖼️ Image Classification
- **Cancer Detection**: Identifies breast cancer, lung cancer, or normal tissue from medical images
- **Deep Learning**: CNN-based model with BatchNormalization and Dropout
- **High Accuracy**: Trained on medical imaging datasets
- **Multiple Formats**: Supports JPEG, PNG, BMP, WebP

### 🩺 Clinical Risk Assessment
- **Risk Prediction**: Evaluates patient health risk based on clinical parameters
- **Feature Analysis**: Age, BMI, blood work, lifestyle factors
- **Confidence Scores**: Provides probability-based risk assessment
- **Fast Inference**: Optimized Random Forest classifier

### 🚀 Production Ready
- **FastAPI Framework**: Modern, fast, async API
- **Centralized Config**: Easy deployment configuration management
- **Comprehensive Logging**: Request tracking and error monitoring
- **Input Validation**: Pydantic models with medical range constraints
- **CORS Support**: Cross-origin resource sharing configured
- **Error Handling**: Graceful error responses with detailed messages

## 🏗️ Architecture

```
┌─────────────┐
│   Client    │
│  (Frontend) │
└──────┬──────┘
       │ HTTP/HTTPS
       ▼
┌─────────────────┐
│   FastAPI       │
│   Backend       │
├─────────────────┤
│ • CORS          │
│ • Validation    │
│ • Logging       │
└────┬───────┬────┘
     │       │
     ▼       ▼
┌─────────┐ ┌──────────────┐
│  Image  │ │   Clinical   │
│  Model  │ │    Model     │
│ (CNN)   │ │ (RandomF)    │
└─────────┘ └──────────────┘
```

## 📁 Project Structure

```
backend/
├── main.py                      # FastAPI application entry point
├── config.py                    # Centralized configuration management
├── requirements.txt             # Python dependencies (pip)
├── pyproject.toml              # Project metadata (uv/pip)
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
│
├── models/                     # Trained ML models
│   ├── image_model.h5         # CNN for image classification
│   └── clinical_model.pkl     # Random Forest for risk assessment
│
└── training/                   # Model training scripts
    ├── image_training/
    │   ├── train_image_model.py
    │   └── dataset/           # Image dataset (not included)
    │       ├── breast_cancer/
    │       ├── lung_cancer/
    │       └── normal/
    │
    └── clinical_training/
        └── train_clinical_model.py
```

## 🛠️ Installation

### Prerequisites

- **Python**: 3.12+ (recommended) or 3.10+
- **pip** or **uv** package manager
- **Git**: For version control

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd backend
```

### Step 2: Create Virtual Environment

**Using venv:**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

**Using uv (faster):**
```bash
uv venv
.venv\Scripts\activate
```

### Step 3: Install Dependencies

**Using pip:**
```bash
pip install -r requirements.txt
```

**Using uv:**
```bash
uv pip install -r requirements.txt
# OR
uv sync
```

### Step 4: Set Up Environment

```bash
# Copy environment template
copy .env.example .env

# Edit .env with your settings (optional for development)
```

## ⚙️ Configuration

All configuration is managed through [config.py](config.py). You can override settings using environment variables.

### Key Configuration Sections

#### API Settings
```python
API_HOST=0.0.0.0          # Server host
API_PORT=8000             # Server port
CORS_ORIGINS=*            # Allowed origins (comma-separated)
```

#### Model Settings
- Image size: 224x224 pixels
- Classes: breast_cancer, lung_cancer, normal
- Clinical features: 8 parameters (age, gender, BMI, etc.)

#### Logging
```python
LOG_LEVEL=INFO            # DEBUG, INFO, WARNING, ERROR
LOG_TO_FILE=false         # Enable file logging
```

### Environment-Specific Settings

**Development:**
```bash
ENVIRONMENT=development
API_RELOAD=true
LOG_LEVEL=DEBUG
```

**Production:**
```bash
ENVIRONMENT=production
API_RELOAD=false
LOG_LEVEL=WARNING
CORS_ORIGINS=https://yourdomain.com
```

## 🚀 Usage

### Start Development Server

```bash
# Method 1: Direct Python
python main.py

# Method 2: Uvicorn CLI
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Method 3: Using uv
uv run python main.py
```

The API will be available at:
- **Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Quick Health Check

```bash
curl http://localhost:8000/health
```

## 📡 API Endpoints

### 1. Root / Health Check

**GET** `/`

Returns service status and model availability.

**Response:**
```json
{
  "status": "running",
  "service": "Hospital AI Backend",
  "version": "1.0.0",
  "environment": "development",
  "models_loaded": true
}
```

### 2. Detailed Health Check

**GET** `/health`

**Response:**
```json
{
  "status": "healthy",
  "image_model": true,
  "clinical_model": true
}
```

### 3. Image Prediction

**POST** `/predict-image`

Predicts cancer type from medical image.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (image file, max 10MB)

**Supported formats:** JPEG, PNG, BMP, WebP

**Example (cURL):**
```bash
curl -X POST http://localhost:8000/predict-image \
  -F "file=@scan.jpg"
```

**Example (Python):**
```python
import requests

with open("scan.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict-image",
        files={"file": f}
    )
print(response.json())
```

**Response:**
```json
{
  "prediction": "lung_cancer",
  "confidence": 0.923
}
```

### 4. Clinical Risk Assessment

**POST** `/predict-clinical`

Predicts health risk from clinical parameters.

**Request Body:**
```json
{
  "age": 55,
  "gender": 1,
  "smoking": 1,
  "alcohol": 0,
  "bmi": 27.5,
  "wbc": 10.2,
  "hemoglobin": 12.8,
  "platelets": 310
}
```

**Field Descriptions:**
- `age`: 0-120 years
- `gender`: 1=Male, 0=Female
- `smoking`: 1=Yes, 0=No
- `alcohol`: 1=Yes, 0=No
- `bmi`: Body Mass Index (10-100)
- `wbc`: White Blood Cell count (0-50)
- `hemoglobin`: g/dL (0-25)
- `platelets`: Count (0-1,000,000)

**Example (cURL):**
```bash
curl -X POST http://localhost:8000/predict-clinical \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "gender": 1,
    "smoking": 1,
    "alcohol": 0,
    "bmi": 27.5,
    "wbc": 10.2,
    "hemoglobin": 12.8,
    "platelets": 310
  }'
```

**Response:**
```json
{
  "prediction": "High Risk",
  "confidence": 0.87
}
```

## 🎓 Training Models

### Train Image Classification Model

**Requirements:**
1. Prepare dataset in `training/image_training/dataset/`:
   ```
   dataset/
   ├── breast_cancer/
   │   ├── image1.jpg
   │   └── image2.jpg
   ├── lung_cancer/
   │   └── ...
   └── normal/
       └── ...
   ```

2. Run training:
   ```bash
   cd training/image_training
   python train_image_model.py
   ```

**Features:**
- CNN with BatchNormalization
- Data augmentation (rotation, zoom, flip)
- Early stopping
- Model checkpointing
- Learning rate reduction

**Output:** `models/image_model.h5`

### Train Clinical Risk Model

**Requirements:**
1. Prepare clinical data CSV or use sample data
2. Run training:
   ```bash
   cd training/clinical_training
   python train_clinical_model.py
   ```

**Features:**
- Random Forest Classifier
- Feature importance analysis
- Classification report
- Stratified train/test split

**Output:** `models/clinical_model.pkl`

## 🌐 Deployment

### Docker Deployment (Recommended)

**Create Dockerfile:**
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create models directory
RUN mkdir -p models

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and run:**
```bash
docker build -t hospital-ai-backend .
docker run -p 8000:8000 -v ./models:/app/models hospital-ai-backend
```

### Cloud Deployment

#### AWS EC2
```bash
# Install dependencies
sudo yum update -y
sudo yum install python3-pip -y

# Clone and setup
git clone <repo>
cd backend
pip3 install -r requirements.txt

# Run with systemd or PM2
```

#### Heroku
```bash
# Create Procfile
echo "web: uvicorn main:app --host 0.0.0.0 --port \$PORT" > Procfile

# Deploy
heroku create hospital-ai-backend
git push heroku main
```

#### Railway / Render
- Connect GitHub repository
- Set build command: `pip install -r requirements.txt`
- Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## 🧪 Testing

### Manual Testing

**Test image endpoint:**
```bash
# Using test image
curl -X POST http://localhost:8000/predict-image \
  -F "file=@test_images/sample.jpg"
```

**Test clinical endpoint:**
```bash
curl -X POST http://localhost:8000/predict-clinical \
  -H "Content-Type: application/json" \
  -d @test_data/clinical_sample.json
```

### Load Testing

```bash
# Install wrk
# Test endpoint performance
wrk -t4 -c100 -d30s http://localhost:8000/
```

## 🔧 Troubleshooting

### Models Not Loading

**Problem:** `Models not loaded: FileNotFoundError`

**Solution:**
1. Train models first (see [Training Models](#training-models))
2. Ensure `models/` directory exists
3. Check file paths in [config.py](config.py)

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'fastapi'`

**Solution:**
```bash
pip install -r requirements.txt
```

### Port Already in Use

**Problem:** `[Errno 48] Address already in use`

**Solution:**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :8000
kill -9 <PID>

# Or use different port
API_PORT=8001 python main.py
```

### CORS Errors

**Problem:** Browser blocks requests

**Solution:**
Update [config.py](config.py) or set environment:
```bash
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
```

### TensorFlow Warnings

**Problem:** GPU warnings or performance messages

**Solution:**
```bash
# Suppress warnings
export TF_CPP_MIN_LOG_LEVEL=2

# Limit threads
export TF_NUM_THREADS=4
```

## 📊 Performance

- **Image Prediction**: ~200-500ms per request
- **Clinical Prediction**: ~10-50ms per request
- **Throughput**: 100+ requests/sec (varies by hardware)
- **Memory Usage**: ~2GB with models loaded

## 🔒 Security Considerations

⚠️ **Production Checklist:**

- [ ] Update CORS origins (remove `*`)
- [ ] Add authentication (JWT, API keys)
- [ ] Enable HTTPS/TLS
- [ ] Rate limiting
- [ ] Input sanitization
- [ ] Secure model files
- [ ] Environment secrets management
- [ ] Regular security updates

## 📝 License

[Add your license here]

## 👥 Contributing

[Add contribution guidelines]

## 📧 Contact

[Add contact information]

---

**Built with:** FastAPI • TensorFlow • Scikit-learn • Python 3.12

**Last Updated:** February 6, 2026
