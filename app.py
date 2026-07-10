import streamlit as st
import pandas as pd
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fraud Detector",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
        * {
            margin: 0;
            padding: 0;
        }
        
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .main {
            background: #f8f9fa;
        }
        
        .stContainer {
            padding: 0;
        }
        
        .header-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px 20px;
            text-align: center;
            color: white;
            margin-bottom: 30px;
            border-radius: 0 0 20px 20px;
        }
        
        .header-title {
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .header-subtitle {
            font-size: 16px;
            opacity: 0.9;
            letter-spacing: 0.5px;
        }
        
        .prediction-card {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        
        .select-label {
            font-weight: 600;
            color: #333;
            font-size: 14px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        
        .stSelectbox, .stMultiSelect {
            border-radius: 8px !important;
        }
        
        .predict-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            font-weight: 700 !important;
            border: none !important;
            padding: 12px 40px !important;
            border-radius: 8px !important;
            cursor: pointer !important;
            font-size: 16px !important;
            transition: all 0.3s ease !important;
            width: 100%;
            margin-top: 20px;
        }
        
        .predict-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4) !important;
        }
        
        .fraud-result {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            font-size: 24px;
            font-weight: 700;
            margin-top: 20px;
            box-shadow: 0 4px 15px rgba(245, 87, 108, 0.3);
        }
        
        .safe-result {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            font-size: 24px;
            font-weight: 700;
            margin-top: 20px;
            box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
        }
        
        .confidence-bar {
            margin: 15px 0;
            padding: 15px;
            background: #f0f0f0;
            border-radius: 8px;
            font-size: 14px;
        }
        
        .loading {
            text-align: center;
            color: #667eea;
            font-weight: 600;
        }
        
        hr {
            margin: 30px 0;
            border: none;
            height: 1px;
            background: linear-gradient(to right, transparent, #ddd, transparent);
        }
        
        .info-box {
            background: #e7f3ff;
            padding: 15px;
            border-left: 4px solid #667eea;
            border-radius: 5px;
            margin: 15px 0;
            font-size: 13px;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv('new_data.xls')
    # Only drop unnecessary index and target columns, keep Time for model
    df = df.drop(['Unnamed: 0', 'Detection'], axis=1, errors='ignore')
    return df

model = load_model()
data = load_data()

# Header
st.markdown("""
    <div class="header-container">
        <div class="header-title"> Credit Card Fraud Detector</div>
        <div class="header-subtitle">AI-Powered Fraud Detection System</div>
    </div>
""", unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<p class="select-label"> Select Transaction</p>', unsafe_allow_html=True)
    
    # Create transaction identifier
    transaction_ids = [f"Transaction {i+1}" for i in range(len(data))]
    selected_transaction = st.selectbox(
        "Choose a transaction to analyze:",
        options=transaction_ids,
        label_visibility="collapsed",
        key="transaction_select"
    )
    
    # Get selected row index
    selected_idx = int(selected_transaction.split()[-1]) - 1
    selected_row = data.iloc[selected_idx].to_dict()

with col2:
    st.markdown('<p class="select-label"> Amount Range</p>', unsafe_allow_html=True)
    amount = st.number_input(
        "Transaction amount:",
        min_value=0.0,
        max_value=float(data['Amount'].max()),
        value=float(data.iloc[selected_idx]['Amount']),
        step=0.1,
        label_visibility="collapsed"
    )

# Display selected transaction data
st.markdown('<hr>', unsafe_allow_html=True)

st.markdown('<p class="select-label"> Transaction Details</p>', unsafe_allow_html=True)

# Show all features except Time (hide Time but use it for predictions)
display_features = [col for col in data.columns if col != 'Time']
feature_cols = st.columns(6)

for idx, feature in enumerate(display_features):
    col_idx = idx % 6
    with feature_cols[col_idx]:
        if feature == 'Amount':
            # Show the customized amount from input
            st.metric(
                label=feature,
                value=f"{amount:.2f}",
                delta=None
            )
        else:
            value = data.iloc[selected_idx][feature]
            st.metric(
                label=feature,
                value=f"{value:.3f}",
                delta=None
            )

# Prepare prediction
prediction_data = data.iloc[selected_idx].copy()
prediction_data['Amount'] = amount

# Predict button
if st.button("🔍 Analyze Transaction", key="predict_btn", use_container_width=True):
    with st.spinner("Analyzing..."):
        # Get model prediction - ensure correct feature order
        X = prediction_data.values.reshape(1, -1)
        
        try:
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0]
            
            # Display result
            if prediction == 1:
                st.markdown(
                    f'<div class="fraud-result"> FRAUD DETECTED<br><small>Risk Level: {probability[1]*100:.1f}%</small></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="safe-result"> TRANSACTION SAFE<br><small>Confidence: {probability[0]*100:.1f}%</small></div>',
                    unsafe_allow_html=True
                )
            
            # Confidence breakdown
            st.markdown('<div class="confidence-bar">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Safe Probability", f"{probability[0]*100:.2f}%")
            with col2:
                st.metric("Fraud Probability", f"{probability[1]*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# Info box
st.markdown("""
    <div class="info-box">
        <strong> How it works:</strong> Select a transaction, adjust the amount if needed, and click "Analyze Transaction" 
        to get real-time fraud detection results powered by machine learning.
    </div>
""", unsafe_allow_html=True)

st.markdown('<hr>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align: center; color: #999; font-size: 12px; padding: 20px;">
        <p>Credit Card Fraud Detection System © 2024 | Powered by Machine Learning</p>
    </div>
""", unsafe_allow_html=True)
