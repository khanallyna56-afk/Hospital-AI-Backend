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

## 📚 Additional Guides

- [UV Package Manager Guide](UV_GUIDE.md) - Complete uv usage reference
- [Migration Guide: pip → uv](MIGRATION_PIP_TO_UV.md) - Switch from pip to uv
- [Quick Start Guide](QUICKSTART_COMPLETE.md) - Get running in 5 minutes
- [Frontend Documentation](frontend/README.md) - Streamlit app guide
- [Changes Log](CHANGES.md) - Version history and updates

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

### 🎯 Combined Risk Assessment (NEW!)
- **Comprehensive Analysis**: Combines both image and clinical data for holistic risk evaluation
- **Risk Meter**: Visual 0-100 risk score with color-coded levels
- **Smart Recommendations**: Context-aware health guidance based on findings
- **Weighted Scoring**: Balances imaging (60%) and clinical (40%) insights

### 💬 AI Doctor Agent (NEW!)
- **LangChain Integration**: Powered by GPT-4 for intelligent medical conversations
- **Contextual Advice**: Discusses your specific assessment results
- **Medical Guidance**: Explains findings in plain language
- **Session Management**: Maintains conversation history and context
- **Educational Focus**: Provides information while emphasizing professional consultation

### 🌐 Streamlit Frontend (NEW!)
- **Interactive UI**: User-friendly web interface for complete assessment
- **Visual Analytics**: Risk gauges, charts, and detailed breakdowns
- **Real-time Chat**: Integrated AI doctor consultation interface
- **Responsive Design**: Works on desktop and tablet devices
- **Status Monitoring**: Live API and model health indicators

### 🚀 Production Ready
- **FastAPI Framework**: Modern, fast, async API
- **Centralized Config**: Easy deployment configuration management
- **Comprehensive Logging**: Request tracking and error monitoring
- **Input Validation**: Pydantic models with medical range constraints
- **CORS Support**: Cross-origin resource sharing configured
- **Error Handling**: Graceful error responses with detailed messages

## 🏗️ Architecture

```
┌──────────────────┐
│   Streamlit UI   │  ← NEW! Interactive Frontend
│  (Frontend App)  │
└────────┬─────────┘
         │ HTTP/REST
         ▼
┌─────────────────────────┐
│   FastAPI Backend       │
├─────────────────────────┤
│ • CORS                  │
│ • Validation            │
│ • Logging               │
│ • Combined Predictions  │  ← NEW!
│ • AI Doctor Agent       │  ← NEW!
└────┬─────────┬──────┬───┘
     │         │      │
     │         │      └──────────┐
     ▼         ▼                 ▼
┌──────────┐ ┌──────────┐  ┌─────────────┐
│  Image   │ │ Clinical │  │  LangChain  │  ← NEW!
│  Model   │ │  Model   │  │  + GPT-4    │
│  (CNN)   │ │ (RF)     │  │ AI Agent    │
└──────────┘ └──────────┘  └─────────────┘
```

## 📁 Project Structure

```
Hospital-AI-Backend/
├── main.py                      # FastAPI application entry point (UPDATED)
├── config.py                    # Centralized configuration management
├── start_app.py                # Quick start script for both services (NEW!)
├── pyproject.toml              # Project metadata & dependencies (uv-managed)
├── uv.lock                     # Dependency lock file (uv-generated)
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
│
├── 📚 Documentation
├── README.md                   # Main documentation (UPDATED)
├── QUICKSTART_COMPLETE.md      # 5-minute setup guide (NEW!)
├── UV_GUIDE.md                 # UV package manager guide (NEW!)
├── MIGRATION_PIP_TO_UV.md      # Migration guide from pip (NEW!)
├── CHANGES.md                  # Version history (UPDATED)
│
├── frontend/                   # Streamlit Web App (NEW!)
│   ├── app.py                 # Main Streamlit application
│   └── README.md              # Frontend documentation
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
- **uv** package manager ([Install uv](https://github.com/astral-sh/uv))
- **Git**: For version control

**Why uv?** 
- ⚡ **10-100x faster** than pip
- 🔒 **Deterministic installs** with lock file
- 🎯 **Better dependency resolution**
- 💾 **Disk space efficient** with global cache
- 🛠️ **All-in-one tool** for Python project management

> 📖 **Migrating from pip?** See our [Migration Guide](MIGRATION_PIP_TO_UV.md)

**Install uv (if not already installed):**
```bash
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd backend
```

### Step 2: Create Virtual Environment

**Using uv (recommended):**
```bash
# Create virtual environment
uv venv

# Activate on Windows
.venv\Scripts\activate

# Activate on Linux/Mac
source .venv/bin/activate
```

**Traditional venv (alternative):**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### Step 3: Install Dependencies

**Using uv (recommended):**
```bash
# Install from pyproject.toml
uv sync

# OR install in editable mode
uv pip install -e .
```

**Traditional pip (alternative):**
```bash
pip install -e .
```

> 💡 **New to uv?** Check out our comprehensive [UV Package Manager Guide](UV_GUIDE.md) for tips, tricks, and best practices.

### Step 4: Set Up Environment

```bash
# Copy environment template
copy .env.example .env  # Windows
# OR
cp .env.example .env    # Linux/Mac

# Edit .env and add your OpenAI API key for AI Doctor feature
# Get your key from: https://platform.openai.com/api-keys
```

**Required for AI Doctor Chat:**
```env
OPENAI_API_KEY=sk-your-actual-openai-api-key-here
```

**Optional settings:**
```env
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
ENVIRONMENT=development
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

### Quick Start (Recommended)

Start both backend and frontend with one command:

```bash
python start_app.py
```

This will:
1. Check for required .env file and models
2. Start the FastAPI backend on port 8000
3. Start the Streamlit frontend on port 8501
4. Auto-open the web interface

Access:
- 🌐 **Web App**: http://localhost:8501
- 📡 **API**: http://localhost:8000
- 📚 **API Docs**: http://localhost:8000/docs

### Manual Start

#### Backend Only

```bash
# Method 1: Direct Python
python main.py

# Method 2: Uvicorn CLI
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Method 3: Using uv
uv run python main.py
```

#### Frontend Only

```bash
# From root directory
streamlit run frontend/app.py

# OR from frontend directory
cd frontend
streamlit run app.py
```

The frontend will be available at:
- **Web App**: http://localhost:8501

The backend API will be available at:
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

### 5. Combined Risk Assessment (NEW!)

**POST** `/predict-combined`

Comprehensive risk assessment combining both image analysis and clinical data.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (image) + clinical data fields

**Example (Python):**
```python
import requests

with open("scan.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict-combined",
        files={"file": f},
        data={
            "age": 55,
            "gender": 1,
            "smoking": 1,
            "alcohol": 0,
            "bmi": 27.5,
            "wbc": 10.2,
            "hemoglobin": 12.8,
            "platelets": 310000
        }
    )
print(response.json())
```

**Response:**
```json
{
  "overall_risk_score": 68.5,
  "risk_level": "High Risk",
  "image_prediction": "lung_cancer",
  "image_confidence": 0.923,
  "clinical_prediction": "High Risk",
  "clinical_confidence": 0.87,
  "recommendations": [
    "⚠️ Immediate consultation with a healthcare provider is recommended",
    "Bring all test results and imaging to your appointment",
    "Discuss lung cancer findings with a specialist"
  ]
}
```

**Risk Levels:**
- **Low Risk**: Score 0-30 (Green)
- **Moderate Risk**: Score 30-60 (Orange)
- **High Risk**: Score 60-100 (Red)

### 6. AI Doctor Chat (NEW!)

**POST** `/chat`

Conversational AI agent for medical advice and consultation.

**Request Body:**
```json
{
  "message": "What do my test results mean?",
  "session_id": "session_123"
}
```

**Fields:**
- `message`: User's question or message
- `session_id`: Optional. If omitted, creates new session

**Example (cURL):**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What should I do about my high risk score?",
    "session_id": "session_123"
  }'
```

**Response:**
```json
{
  "response": "Based on your high risk assessment...",
  "session_id": "session_123",
  "timestamp": "2026-02-06T10:30:45.123456"
}
```

**Features:**
- Maintains conversation context per session
- Provides medical guidance based on assessment results
- Explains medical terms in plain language
- Emphasizes professional consultation when needed

### 7. End Chat Session (NEW!)

**DELETE** `/chat/{session_id}`

Ends a chat session and clears conversation history.

**Example:**
```bash
curl -X DELETE http://localhost:8000/chat/session_123
```

**Response:**
```json
{
  "status": "success",
  "message": "Session ended"
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

# Install uv
RUN pip install uv

# Install dependencies
COPY pyproject.toml .
RUN uv pip install --system -e .

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
docker run -p 8000:8000 -v ./models:/app/models -e OPENAI_API_KEY=your_key hospital-ai-backend
```

### Cloud Deployment

#### AWS EC2
```bash
# Install Python and uv
sudo yum update -y
sudo yum install python3-pip -y

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

# Clone and setup
git clone <repo>
cd Hospital-AI-Backend
uv sync

# Run with systemd or PM2
```

#### Heroku
```bash
# Create Procfile
echo "web: uvicorn main:app --host 0.0.0.0 --port \$PORT" > Procfile

# Create runtime.txt for Python version
echo "python-3.12" > runtime.txt

# Deploy
heroku create hospital-ai-backend
heroku config:set OPENAI_API_KEY=your_key
git push heroku main
```

#### Railway / Render
- Connect GitHub repository
- Set build command: `pip install uv && uv sync`
- Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Add environment variable: `OPENAI_API_KEY=your_key`

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
# Using uv (recommended)
uv sync --reinstall

# OR
uv pip install -e . --reinstall

# Traditional pip
pip install -e .
```

### uv Installation Issues

**Problem:** `uv: command not found`

**Solution:**
```bash
# Install uv
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

### Dependency Resolution Errors

**Problem:** Conflicts with pandas or pillow versions

**Solution:**
Streamlit requires specific version ranges:
- `pandas>=2.0.0,<3` (not 3.x)
- `pillow>=10.0.0,<12` (not 12.x)

These constraints are already set in `pyproject.toml`. If you encounter issues:
```bash
# Clear cache and reinstall
uv cache clean
uv sync --reinstall
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
