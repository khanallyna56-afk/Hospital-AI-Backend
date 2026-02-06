"""
Hospital AI Frontend - Streamlit Application
Interactive interface for medical risk assessment and AI doctor consultation
"""

import streamlit as st
import requests
import plotly.graph_objects as go
from PIL import Image
import io
import json

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Hospital AI - Medical Risk Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-meter {
        text-align: center;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    .moderate-risk {
        background-color: #fff3e0;
        border: 2px solid #ff9800;
    }
    .low-risk {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
    }
    .chat-container {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


def create_risk_gauge(risk_score, risk_level):
    """Create a gauge chart for risk visualization"""
    
    # Determine color based on risk level
    if risk_score < 30:
        color = "#4caf50"  # Green
    elif risk_score < 60:
        color = "#ff9800"  # Orange
    else:
        color = "#f44336"  # Red
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Overall Risk Score<br><b>{risk_level}</b>", 'font': {'size': 24}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#e8f5e9'},
                {'range': [30, 60], 'color': '#fff3e0'},
                {'range': [60, 100], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=80, b=20),
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Hospital AI Medical Assistant</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.info("""
        This AI-powered medical assistant provides:
        - **Image Analysis**: Cancer detection from medical images
        - **Clinical Assessment**: Risk evaluation from lab results
        - **Combined Risk Score**: Comprehensive health assessment
        - **AI Doctor Chat**: Get medical advice and guidance
        
        ⚠️ **Important**: This is for informational purposes only. 
        Always consult healthcare professionals for medical decisions.
        """)
        
        st.header("🔧 System Status")
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                st.success("✅ API Connected")
                st.write(f"Image Model: {'✅' if health['image_model'] else '❌'}")
                st.write(f"Clinical Model: {'✅' if health['clinical_model'] else '❌'}")
            else:
                st.error("⚠️ API Issues")
        except Exception as e:
            st.error(f"❌ API Offline")
            st.caption(str(e))
    
    # Main content - Two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📊 Medical Assessment")
        
        # File upload
        st.subheader("1️⃣ Upload Medical Image")
        uploaded_file = st.file_uploader(
            "Choose a medical image (X-ray, CT scan, etc.)",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a medical image for cancer detection"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Clinical data input
        st.subheader("2️⃣ Enter Clinical Data")
        
        with st.form("clinical_data_form"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                age = st.number_input("Age", min_value=0, max_value=120, value=45, step=1)
                gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
                smoking = st.selectbox("Smoking", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                alcohol = st.selectbox("Alcohol Consumption", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            
            with col_b:
                bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
                wbc = st.number_input("WBC Count (×10³/μL)", min_value=0.0, max_value=50.0, value=7.5, step=0.1)
                hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=25.0, value=14.0, step=0.1)
                platelets = st.number_input("Platelets (×10³/μL)", min_value=0, max_value=1000000, value=250000, step=1000)
            
            submit_button = st.form_submit_button("🔍 Analyze Complete Risk Profile", use_container_width=True)
        
        if submit_button:
            if not uploaded_file:
                st.error("Please upload a medical image first!")
            else:
                with st.spinner("Analyzing your medical data... Please wait."):
                    try:
                        # Prepare form data
                        files = {
                            'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                        }
                        data = {
                            'age': age,
                            'gender': gender,
                            'smoking': smoking,
                            'alcohol': alcohol,
                            'bmi': bmi,
                            'wbc': wbc,
                            'hemoglobin': hemoglobin,
                            'platelets': platelets
                        }
                        
                        # Make API request
                        response = requests.post(
                            f"{API_BASE_URL}/predict-combined",
                            files=files,
                            data=data,
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.prediction_result = result
                            st.success("✅ Analysis Complete!")
                        else:
                            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                    
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to API. Please ensure the backend is running.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.header("📈 Risk Assessment Results")
        
        if st.session_state.prediction_result:
            result = st.session_state.prediction_result
            
            # Risk Gauge
            st.plotly_chart(
                create_risk_gauge(result['overall_risk_score'], result['risk_level']),
                use_container_width=True
            )
            
            # Detailed results
            st.subheader("Detailed Analysis")
            
            col_x, col_y = st.columns(2)
            
            with col_x:
                st.metric(
                    "Image Analysis",
                    result['image_prediction'].replace('_', ' ').title(),
                    f"{result['image_confidence']*100:.1f}% confidence"
                )
            
            with col_y:
                st.metric(
                    "Clinical Assessment",
                    result['clinical_prediction'],
                    f"{result['clinical_confidence']*100:.1f}% confidence"
                )
            
            # Recommendations
            st.subheader("📋 Recommendations")
            risk_class = "low-risk" if result['overall_risk_score'] < 30 else \
                        "moderate-risk" if result['overall_risk_score'] < 60 else "high-risk"
            
            st.markdown(f'<div class="risk-meter {risk_class}">', unsafe_allow_html=True)
            for rec in result['recommendations']:
                st.write(f"• {rec}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Auto-populate chat with context
            if result['overall_risk_score'] > 30:
                context_message = f"""Based on my assessment:
- Risk Score: {result['overall_risk_score']:.1f}
- Image Analysis: {result['image_prediction'].replace('_', ' ')}
- Clinical Risk: {result['clinical_prediction']}

What should I be concerned about and what are my next steps?"""
                
                if st.button("💬 Discuss Results with AI Doctor"):
                    st.session_state.chat_history = []
                    st.session_state.initial_message = context_message
                    st.rerun()
        else:
            st.info("👆 Complete the medical assessment form to see your risk analysis here.")
    
    # Chat section - Full width
    st.markdown("---")
    st.header("💬 Chat with AI Doctor")
    
    col_chat1, col_chat2 = st.columns([2, 1])
    
    with col_chat1:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**🩺 AI Doctor:** {message['content']}")
            st.markdown("---")
        
        # Check for initial message
        if 'initial_message' in st.session_state:
            initial_msg = st.session_state.initial_message
            del st.session_state.initial_message
            
            # Send to chat
            try:
                response = requests.post(
                    f"{API_BASE_URL}/chat",
                    json={
                        "message": initial_msg,
                        "session_id": st.session_state.session_id
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    chat_response = response.json()
                    st.session_state.session_id = chat_response['session_id']
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': initial_msg
                    })
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': chat_response['response']
                    })
                    st.rerun()
            except Exception as e:
                st.error(f"Chat error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_chat2:
        st.info("""
        **Ask the AI Doctor about:**
        - Understanding your results
        - Preventive measures
        - Lifestyle recommendations
        - When to seek medical care
        - General health questions
        """)
        
        if st.button("🗑️ Clear Chat History"):
            if st.session_state.session_id:
                try:
                    requests.delete(f"{API_BASE_URL}/chat/{st.session_state.session_id}")
                except:
                    pass
            st.session_state.chat_history = []
            st.session_state.session_id = None
            st.rerun()
    
    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        user_message = st.text_input(
            "Your message:",
            placeholder="Ask me anything about your health assessment...",
            label_visibility="collapsed"
        )
        send_button = st.form_submit_button("Send 📤")
        
        if send_button and user_message:
            with st.spinner("AI Doctor is thinking..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/chat",
                        json={
                            "message": user_message,
                            "session_id": st.session_state.session_id
                        },
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        chat_response = response.json()
                        st.session_state.session_id = chat_response['session_id']
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'role': 'user',
                            'content': user_message
                        })
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': chat_response['response']
                        })
                        
                        st.rerun()
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Ensure backend is running and OPENAI_API_KEY is set.")
                except Exception as e:
                    st.error(f"❌ Chat error: {str(e)}")


if __name__ == "__main__":
    main()
