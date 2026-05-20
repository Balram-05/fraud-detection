import streamlit as st
import requests
import numpy as np

BACKEND_URL = "http://127.0.0.1:8000"  # Update to your live Render backend URL if testing in the cloud

st.set_page_config(page_title="Risk Operations Command Center", layout="wide")

st.title("🛡️ Enterprise Fraud Detection Sandbox")
st.markdown("Adjust the transaction parameters below to observe how the underlying machine learning model evaluates risk profiles in real time.")

st.sidebar.header("Transaction Vector Controls")

# Real, interactive variables for the user to manipulate
amount = st.sidebar.slider("Transaction Amount ($)", min_value=1.0, max_value=5000.0, value=85.0, step=0.5)
v4 = st.sidebar.slider("V4 (Feature Weight: High Risk Indication)", min_value=-5.0, max_value=10.0, value=0.0, step=0.1)
v11 = st.sidebar.slider("V11 (Feature Weight: Behavior Anomaly)", min_value=-5.0, max_value=10.0, value=0.0, step=0.1)
v12 = st.sidebar.slider("V12 (Feature Weight: Security Match)", min_value=-15.0, max_value=5.0, value=0.0, step=0.1)
v14 = st.sidebar.slider("V14 (Feature Weight: Integrity Anomaly)", min_value=-15.0, max_value=5.0, value=0.0, step=0.1)

# Preset Quick-Configurations to help the user test instantly
st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Preset Scenarios")
if st.sidebar.button("Set Legitimate Baseline"):
    st.info("Adjusting sliders to standard spending baseline indicators...")
    # These values typically look standard to a PCA model
    amount, v4, v11, v12, v14 = 50.0, -0.5, -0.3, 0.5, 0.6

if st.sidebar.button("Set High-Risk Malicious Anomaly"):
    st.warning("Adjusting sliders to classic fraudulent signature profiles...")
    # Highly skewed values that typically trigger anomaly thresholds
    amount, v4, v11, v12, v14 = 2500.0, 6.5, 5.0, -8.0, -9.5

# 1. Generate exactly 14 passive baseline variables (29 total expected - 15 explicit slots)
np.random.seed(42)  
base_features = list(np.random.normal(0, 0.2, 14))

# 2. Map the array cleanly (14 elements + 14 elements + 1 element = 29 total features)
full_payload_features = [0.0, 0.0, 0.0, v4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, v11, v12, 0.0, v14] + base_features + [amount]
# Main Dashboard Interface Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Active Operational Payload Matrix")
    st.json({
        "Transaction Amount": amount,
        "V4 (Risk Factor)": v4,
        "V11 (Behavioral)": v11,
        "V12 (Security)": v12,
        "V14 (Integrity)": v14,
        "Passive Features (V1-V28 Hidden Context)": "Compiling 24 balanced baseline dimensions..."
    })

with col2:
    st.subheader("Real-Time Machine Learning Verdict")
    
    if st.button("Evaluate Live Transaction Profile", type="primary"):
        try:
            # 1. Fetch the continuous model score probability path
            score_res = requests.get(f"{BACKEND_URL}/score", params={"f": full_payload_features})
            # 2. Fetch the hard classification action response
            predict_res = requests.post(f"{BACKEND_URL}/predict", json={"features": full_payload_features})
            
            if score_res.status_code == 200 and predict_res.status_code == 200:
                probability = score_res.json()["fraud_probability"] * 100
                prediction = predict_res.json()["fraud_prediction"]
                
                # Dynamic graphic rendering based on the shifting score metric output
                st.metric(label="Calculated Model Fraud Risk Score", value=f"{probability:.2f}%")
                
                if prediction == 1 or probability > 50.0:
                    st.error("🚨 TRANSACTION MULTI-FACTOR ALERT: DECLINED")
                    st.markdown("**Reasoning:** System vector deviations crossed compliance tolerance thresholds. Anomalous signature identified.")
                else:
                    st.success("✅ TRANSACTION AUTHORIZED: APPROVED")
                    st.markdown("**Reasoning:** Spending metrics track safely within established customer baseline variables.")
            else:
                st.error(f"Error connecting to infrastructure: API returned status code {predict_res.status_code}")
                
        except Exception as e:
            st.error(f"Failed to communicate with running backend server. Details: {e}")