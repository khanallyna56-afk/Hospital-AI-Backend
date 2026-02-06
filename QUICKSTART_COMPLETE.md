# 🚀 Hospital AI - Quick Start Guide

Get your complete medical AI system running in 5 minutes!

## ⚡ Prerequisites

Before you begin, ensure you have:

- ✅ **Python 3.12+** installed ([Download](https://www.python.org/downloads/))
- ✅ **uv package manager** ([Install uv](https://github.com/astral-sh/uv))
- ✅ **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))
- ✅ **Git** installed (optional, for cloning)

### Install uv (if not already installed)

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

> 💡 **Coming from pip?** Check out our [Migration Guide](MIGRATION_PIP_TO_UV.md) for a smooth transition!

## 📥 Step 1: Installation

### Clone or Download Repository

```bash
# Using Git
git clone <repository-url>
cd Hospital-AI-Backend

# OR download and extract the ZIP file
```

### Install Dependencies

```bash
# Using uv (recommended - much faster!)
uv sync

# OR install in editable mode
uv pip install -e .

# Traditional pip (slower alternative)
pip install -e .
```

## 🔑 Step 2: Configure API Key

### Create Environment File

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

### Add Your OpenAI API Key

Open `.env` in any text editor and update:

```env
OPENAI_API_KEY=sk-your-actual-api-key-here
```

**Where to get your API key:**
1. Go to https://platform.openai.com/api-keys
2. Sign up or log in
3. Create a new API key
4. Copy and paste it into your `.env` file

## 🤖 Step 3: Train Models

### Train Clinical Model (Fast - ~10 seconds)

```bash
python training/clinical_training/train_clinical_model.py
```

**Expected output:**
```
✓ Training complete
Model Accuracy: 0.96+
✅ Clinical model saved at: models/clinical_model.pkl
```

### Train Image Model (Optional - requires dataset)

**Note:** Image model training requires a dataset. If you don't have one:

1. **Option A:** Use a pre-trained model (if provided)
2. **Option B:** Prepare your dataset:
   ```
   training/image_training/dataset/
   ├── breast_cancer/
   ├── lung_cancer/
   └── normal/
   ```

3. **Option C:** Skip for now and use clinical-only features

```bash
# If you have a dataset
python training/image_training/train_image_model.py
```

## 🎯 Step 4: Start the Application

### Start Backend (Terminal 1)

```bash
# Using uv (recommended - auto-activates venv)
uv run python main.py

# Or activate venv manually first
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

python main.py
```

### Start Frontend (Terminal 2)

```bash
# Using uv (recommended - auto-activates venv)
uv run streamlit run frontend/app.py

# Or with activated venv
streamlit run frontend/app.py
```

## 🌐 Step 5: Access the Application

### Open Your Browser

The app should open automatically at:
- **🎨 Frontend Web App:** http://localhost:8501
- **📡 Backend API:** http://localhost:8000
- **📚 API Documentation:** http://localhost:8000/docs

### What You'll See

✅ **Green status indicators** = Everything working!
❌ **Red status indicators** = Check the troubleshooting section below

## 🎓 Step 6: Try It Out!

### Complete Medical Assessment

1. **Upload a Medical Image**
   - Use any medical scan image (JPEG, PNG)
   - For testing, you can use sample X-ray images from the web

2. **Enter Clinical Data**
   - Age: 55
   - Gender: Male
   - Smoking: Yes
   - Alcohol: No
   - BMI: 27.5
   - WBC: 10.2
   - Hemoglobin: 14.0
   - Platelets: 250000

3. **Click "Analyze Complete Risk Profile"**

4. **View Your Results**
   - Overall risk score with visual gauge
   - Detailed analysis breakdown
   - Personalized recommendations

### Chat with AI Doctor

1. **Ask Questions** like:
   - "What do these results mean?"
   - "What should I be concerned about?"
   - "What lifestyle changes can I make?"
   - "When should I see a doctor?"

2. **Get Contextual Advice**
   - The AI doctor knows your assessment results
   - Provides clear, compassionate guidance
   - Explains medical terms in plain language

## 🛠️ Troubleshooting

### ❌ "Cannot connect to API"

**Problem:** Frontend can't reach backend

**Solutions:**
```bash
# 1. Check if backend is running
curl http://localhost:8000/health

# 2. Restart backend
python main.py

# 3. Check port availability
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # Linux/Mac
```

### ❌ "AI doctor agent not available"

**Problem:** OpenAI API key not set or invalid

**Solutions:**
1. Check `.env` file exists
2. Verify `OPENAI_API_KEY` is set correctly
3. Test your API key at https://platform.openai.com
4. Restart the backend after updating `.env`

### ❌ "Models not available"

**Problem:** ML models not trained yet

**Solutions:**
```bash
# Train clinical model
python training/clinical_training/train_clinical_model.py

# Check models directory
ls models/  # Should show clinical_model.pkl and image_model.h5
```

### ❌ Import errors or missing packages

**Problem:** Dependencies not installed

**Solutions:**
```bash
# Reinstall all dependencies using uv
uv sync --reinstall

# Traditional pip (slower)
pip install -e . --force-reinstall
```

### ❌ Dependency resolution errors

**Problem:** Conflicts with pandas 3.x or pillow 12.x

**Explanation:** Streamlit requires:
- pandas <3.0 (we use pandas 2.x)
- pillow <12.0 (we use pillow 11.x)

These constraints are already configured in `pyproject.toml`.

**Solution:**
```bash
# Clear cache and sync
uv cache clean
uv sync
```

### ❌ TensorFlow warnings

**Problem:** TensorFlow showing warnings (safe to ignore)

**Solution:** Already suppressed in code, but you can add to `.env`:
```env
TF_CPP_MIN_LOG_LEVEL=3
TF_ENABLE_ONEDNN_OPTS=0
```

## 🎯 Next Steps

### Explore the API

1. Open API documentation: http://localhost:8000/docs
2. Try the interactive endpoints
3. Test with cURL or Postman

### Customize Configuration

Edit `config.py` to adjust:
- Model parameters
- Validation constraints
- API settings
- Logging levels

### Deploy to Production

See [README.md](README.md) for:
- Docker deployment
- Cloud hosting (AWS, Azure, GCP)
- Security best practices
- Performance optimization

## 📚 Learn More

- **Full Documentation:** [README.md](README.md)
- **Frontend Guide:** [frontend/README.md](frontend/README.md)
- **API Reference:** http://localhost:8000/docs
- **Configuration:** [config.py](config.py)

## 🆘 Getting Help

### Common Use Cases

**Medical Image Only:**
```bash
curl -X POST http://localhost:8000/predict-image \
  -F "file=@scan.jpg"
```

**Clinical Data Only:**
```bash
curl -X POST http://localhost:8000/predict-clinical \
  -H "Content-Type: application/json" \
  -d '{"age": 55, "gender": 1, "smoking": 1, "alcohol": 0, "bmi": 27.5, "wbc": 10.2, "hemoglobin": 14.0, "platelets": 250000}'
```

**Combined Assessment:**
Use the Streamlit web interface for the best experience!

## ⚠️ Important Reminders

- 🔒 **Never commit your `.env` file** (it contains your API key)
- 📋 **This is for educational purposes** - not for actual medical diagnosis
- 👨‍⚕️ **Always consult healthcare professionals** for medical decisions
- 💰 **OpenAI API usage** incurs costs - monitor your usage at https://platform.openai.com/usage

## ✅ Success Checklist

- [ ] Python 3.12+ installed
- [ ] uv package manager installed
- [ ] Dependencies installed (`uv sync`)
- [ ] `.env` file created with valid `OPENAI_API_KEY`
- [ ] Clinical model trained successfully
- [ ] Backend running on port 8000
- [ ] Frontend running on port 8501
- [ ] Can access both services in browser
- [ ] Models loaded (green checkmarks in UI)
- [ ] Can upload image and enter clinical data
- [ ] Can chat with AI doctor

**All checked?** 🎉 **You're ready to go!**

---

**Happy diagnosing! 🏥✨**

For detailed documentation, see [README.md](README.md)
