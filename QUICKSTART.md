# 🚀 Quick Start Guide

## Hospital AI Backend - Get Running in 5 Minutes

### Step 1: Setup Environment (1 min)

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies (2 min)

```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
python -c "import fastapi; print('✓ FastAPI installed')"
```

### Step 3: Setup Configuration (30 sec)

```bash
# Copy environment template
copy .env.example .env

# (Optional) Edit .env file for custom settings
# Default settings work fine for development
```

### Step 4: Run the Server (30 sec)

```bash
# Start the API server
python main.py

# You should see:
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 5: Test It! (1 min)

**Open your browser:**
- Main docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

**Or test with curl:**

```bash
# Health check
curl http://localhost:8000/health

# Test image prediction (if model exists)
curl -X POST http://localhost:8000/predict-image \
  -F "file=@your_image.jpg"

# Test clinical prediction
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

---

## ⚠️ Important Notes

### No Models Yet?

If you see "models not loaded", you need to train them first:

```bash
# Train clinical model (quick - sample data)
cd training/clinical_training
python train_clinical_model.py
cd ../..

# Train image model (requires dataset)
# 1. Add your images to training/image_training/dataset/
# 2. Organize in folders: breast_cancer/, lung_cancer/, normal/
# 3. Run training:
cd training/image_training
python train_image_model.py
cd ../..
```

### Common Issues

**Port 8000 in use?**
```bash
# Use different port
API_PORT=8001 python main.py
```

**Import errors?**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Config errors?**
```bash
# Make sure you're in the backend directory
cd backend
python main.py
```

---

## 📚 Next Steps

1. **Read Full Docs**: See [README.md](README.md) for complete documentation
2. **Train Models**: Follow training guide in README
3. **Test Endpoints**: Use interactive docs at `/docs`
4. **Configure for Production**: Update [config.py](config.py) and `.env`

---

## 🎯 Development Workflow

```bash
# 1. Make changes to code
# 2. Server auto-reloads (if reload=true)
# 3. Test at http://localhost:8000/docs
# 4. Check logs in terminal
```

## 🔥 Hot Tips

- **Interactive Testing**: Use `/docs` - it's powered by Swagger UI
- **Configuration**: All settings in [config.py](config.py) - no hardcoded values!
- **Logging**: Check terminal for request logs and errors
- **API Keys**: Not required in development mode

---

**Questions?** Check [README.md](README.md) or [Troubleshooting](#common-issues) section

**Ready to Deploy?** See deployment guide in main README
