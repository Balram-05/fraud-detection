import streamlit as st
import requests
import os

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Risk Operations Portal", page_icon="🛡️", layout="wide")

st.title("🛡️ Risk Operations & Fraud Inference Platform")
st.caption("Enterprise Transaction Auditing Terminal | Production Engine v1.0.0")
st.markdown("---")

# 1. Human-Readable Control Inputs
st.subheader("📥 Transaction Audit Parameters")
c1, c2, c3 = st.columns(3)

with c1:
    account_id = st.text_input("Account Identifier", value="ACC-99481-X")
with c2:
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=89.90, step=0.01)
with c3:
    risk_profile = st.selectbox(
        "Simulated Behavioral Risk Profile", 
        ["Standard Customer Pattern", "Verified Malicious Fraud Pattern"]
    )

# 2. Hardcoded baseline vectors straight from your dataset splits
# This bypasses random generation bugs entirely!
if risk_profile == "Standard Customer Pattern":
    # Real dataset row for Class 0 (Normal)
    active_vector = [-1.3598, 0.0727, 2.5363, 1.3781, -0.3383, 0.4623, 0.2395, 0.0986, 0.3637, 0.0907, -0.5516, -0.6178, -0.9913, -0.3111, 1.4681, -0.4704, 0.2079, 0.0257, 0.4039, 0.2514, -0.0183, 0.2778, -0.1104, 0.0669, 0.1285, -0.1891, 0.1335, -0.0211]
else:
    # Real dataset row for Class 1 (Fraud) - Highly anomalous vectors
    active_vector = [-2.3122, 1.9519, -1.6098, 3.9979, -0.5221, -1.4265, -2.5373, 1.3916, -2.7700, -2.7722, 3.2020, -2.8999, -0.5952, -4.2892, 0.3897, -1.1401, -2.8300, -0.0161, 0.4160, 0.1269, 0.5177, -0.0350, -0.4652, 0.3201, 0.0445, 0.1778, 0.2611, -0.1432]

# Add the user-defined amount scaled down as the 29th feature element
scaled_amount = amount / 100.0
full_payload_features = active_vector + [scaled_amount]

st.markdown(" ")

# 3. Render Technical Metadata Panel (Refreshes instantly on dropdown change)
with st.expander("⚙️ Pipeline Technical Metadata Grid (Read-Only Compliance Layer)", expanded=True):
    grid_cols = st.columns(6)
    for i in range(1, 29):
        col_idx = (i - 1) % 6
        with grid_cols[col_idx]:
            st.number_input(f"V{i}", value=float(active_vector[i-1]), disabled=True, key=f"v_view_{i}")
    
    with grid_cols[4]:
        st.number_input("Scaled Amount Feature", value=float(scaled_amount), disabled=True, key="amount_view")

st.markdown("---")

# 4. Assessment Engine Interaction
if st.button("Run Fraud Verification", type="primary", use_container_width=True):
    with st.spinner("Processing network telemetry through backend API..."):
        try:
            # Send payload array to FastAPI backend POST /predict endpoint
            predict_res = requests.post(f"{BACKEND_URL}/predict", json={"features": full_payload_features})
            
            # Send query sequence to GET /score endpoint
            query_params = "&".join([f"f={val}" for val in full_payload_features])
            score_res = requests.get(f"{BACKEND_URL}/score?{query_params}")
            
            if predict_res.status_code == 200 and score_res.status_code == 200:
                prediction = predict_res.json()
                score = score_res.json()
                prob_pct = score["fraud_probability"] * 100
                
                st.subheader("🔍 Real-Time Assessment Results")
                
                # Render UI feedback based on raw model outputs
                if prediction["fraud_prediction"] == 1:
                    st.error("🚨 **TRANSACTION MULTI-FACTOR ALERT: DECLINED**")
                    st.metric(label="Calculated Risk Score", value=f"{prob_pct:.2f}%", delta="CRITICAL FRAUD SIGNATURE", delta_color="inverse")
                    st.warning(f"Action Flag: Routed to Remediation Team. Account associated with transaction temporarily locked.")
                else:
                    st.success("✅ **TRANSACTION CLEAR: APPROVED**")
                    st.metric(label="Calculated Risk Score", value=f"{prob_pct:.2f}%", delta="SAFE GENUINE PATTERN")
                    st.info("Action Flag: Transaction cleared compliance parameters successfully.")
            else:
                st.error("System Error: Backend endpoints responded with invalid status markers.")
        except requests.exceptions.ConnectionError:
            st.error("Connectivity Fault: Frontend failed to reach the FastAPI microservice engine.")