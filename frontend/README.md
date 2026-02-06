# Hospital AI Frontend

A Streamlit-based web application for medical risk assessment using AI.

## Features

- 🖼️ **Medical Image Analysis**: Upload medical images for cancer detection
- 📊 **Clinical Data Assessment**: Input clinical parameters for risk evaluation
- 🎯 **Combined Risk Score**: Get comprehensive health risk assessment with visual gauge
- 💬 **AI Doctor Chat**: Conversational AI agent for medical advice and guidance
- 📈 **Visual Analytics**: Interactive risk meters and detailed breakdowns

## Installation

### 1. Install Dependencies

From the root project directory:

```bash
# Using uv (recommended - faster!)
uv sync

# OR install in editable mode
uv pip install -e .
```

Or install specific frontend requirements:

```bash
# Using uv
uv pip install streamlit plotly requests

# Using pip
pip install streamlit>=1.40.0 plotly>=5.24.0 requests
```

### 2. Set Up Environment Variables

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=your_actual_openai_api_key_here
```

Get your API key from: https://platform.openai.com/api-keys

### 3. Train Models (If Not Already Done)

```bash
# Train the clinical model
python training/clinical_training/train_clinical_model.py

# Train the image model (requires dataset)
python training/image_training/train_image_model.py
```

## Running the Application

### 1. Start the Backend API

From the root directory:

```bash
python main.py
```

The API will start at `http://localhost:8000`

### 2. Start the Streamlit Frontend

In a new terminal, from the root directory:

```bash
streamlit run frontend/app.py
```

Or from the frontend directory:

```bash
cd frontend
streamlit run app.py
```

The web app will open automatically at `http://localhost:8501`

## Usage

### Medical Assessment

1. **Upload Image**: Choose a medical image (X-ray, CT scan)
2. **Enter Clinical Data**: Fill in patient information
   - Age, gender, smoking status, alcohol consumption
   - BMI, WBC count, hemoglobin, platelet count
3. **Analyze**: Click "Analyze Complete Risk Profile"
4. **View Results**: See combined risk score, detailed analysis, and recommendations

### AI Doctor Chat

1. **Start Conversation**: Type your question in the chat input
2. **Get Advice**: Receive personalized medical guidance
3. **Contextual Help**: After assessment, click "Discuss Results with AI Doctor" for targeted advice
4. **Clear History**: Use "Clear Chat History" to start fresh

## Features Overview

### Risk Assessment Components

- **Overall Risk Score**: 0-100 scale with visual gauge
  - 0-30: Low Risk (Green)
  - 30-60: Moderate Risk (Orange)
  - 60-100: High Risk (Red)

- **Image Analysis**: Cancer detection from medical imaging
  - Breast cancer detection
  - Lung cancer detection
  - Normal classification

- **Clinical Assessment**: Risk evaluation from lab results
  - High Risk / Low Risk classification
  - Confidence scores

### AI Doctor Agent

- Powered by LangChain and GPT-4
- Provides educational medical information
- Explains medical terms in plain language
- Offers preventive measures and lifestyle recommendations
- Emphasizes professional consultation when necessary

## API Endpoints Used

- `GET /health` - Check API and model status
- `POST /predict-combined` - Combined prediction endpoint
- `POST /chat` - Chat with AI doctor
- `DELETE /chat/{session_id}` - End chat session

## Configuration

Edit constants in `app.py`:

```python
API_BASE_URL = "http://localhost:8000"  # Backend API URL
```

## Troubleshooting

### Backend Not Connecting

```
❌ Cannot connect to API
```

**Solution**: Ensure the backend is running on port 8000:
```bash
python main.py
```

### AI Chat Not Working

```
AI doctor agent not available
```

**Solution**: Set your OpenAI API key in `.env`:
```
OPENAI_API_KEY=sk-...
```

### Models Not Available

```
Models not available
```

**Solution**: Train the models first:
```bash
python training/clinical_training/train_clinical_model.py
python training/image_training/train_image_model.py
```

## Security Notice

⚠️ **Important**: This application is for educational and informational purposes only. 

- Always consult qualified healthcare professionals for medical decisions
- Do not use for actual medical diagnosis
- Protect patient privacy and data
- Use secure connections in production
- Never expose API keys in code

## Technology Stack

- **Frontend**: Streamlit
- **Visualization**: Plotly
- **Backend**: FastAPI (main.py)
- **AI**: LangChain + OpenAI GPT-4
- **ML Models**: TensorFlow, scikit-learn

## License

See root project LICENSE file.

## Support

For issues and questions, please refer to the main project README.
