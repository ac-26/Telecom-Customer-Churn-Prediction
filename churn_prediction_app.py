import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {padding-top: 3rem;}
    h1 {color: #1f77b4;}
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🔮 Telco Customer Churn Predictor")
st.markdown("### Predict customer churn probability and get actionable retention insights")

# Sidebar - Inputs
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/phone.png", width=80)
    st.title("Customer Information")
    st.markdown("---")
    
    st.subheader("📊 Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    
    st.markdown("---")
    
    st.subheader("📋 Account Information")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", 
                                 ["Electronic check", "Mailed check", 
                                  "Bank transfer (automatic)", "Credit card (automatic)"])
    
    st.markdown("---")
    
    st.subheader("📡 Services")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    if phone_service == "Yes":
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
    else:
        multiple_lines = "No phone service"
    
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    if internet_service != "No":
        online_security = st.selectbox("Online Security", ["Yes", "No"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
    else:
        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"
    
    st.markdown("---")
    
    st.subheader("💰 Charges")
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0, 5.0)
    total_charges = monthly_charges * max(tenure, 1)
    st.info(f"Total Charges: ${total_charges:.2f}")

# Main Prediction Logic
if st.button("🎯 Predict Churn", use_container_width=True):
    # Prepare input
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })

    input_data['AvgMonthlyCharges'] = input_data['TotalCharges'] / np.maximum(input_data['tenure'], 1)

    addon_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    input_data['NumAddOnServices'] = (input_data[addon_cols] == 'Yes').sum(axis=1)

    # Churn risk score (fake model logic)
    risk_score = 0.3
    if contract == "Month-to-month":
        risk_score += 0.25
    if payment_method == "Electronic check":
        risk_score += 0.2
    if senior_citizen == "Yes":
        risk_score += 0.15
    if tenure < 12:
        risk_score += 0.15

    if contract == "Two year":
        risk_score -= 0.2
    if input_data['NumAddOnServices'].iloc[0] >= 4:
        risk_score -= 0.15
    if tenure > 48:
        risk_score -= 0.2

    risk_score = max(0, min(1, risk_score))

    # Risk level
    if risk_score < 0.25:
        risk_level = "Low Risk"
        risk_color = "green"
        risk_icon = "✅"
    elif risk_score < 0.5:
        risk_level = "Moderate Risk"
        risk_color = "orange"
        risk_icon = "⚠️"
    elif risk_score < 0.75:
        risk_level = "High Risk"
        risk_color = "red"
        risk_icon = "🚨"
    else:
        risk_level = "Critical Risk"
        risk_color = "darkred"
        risk_icon = "🆘"

    # Show result only
    st.markdown(f"""
    <div style='text-align: center; padding: 100px 20px; background-color: #f8f9fa; border-radius: 12px;'>
        <h2 style='color: {risk_color}; font-size: 40px;'>{risk_icon} {risk_level}</h2>
        <p style='font-size: 22px;'>Churn Probability: {risk_score*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)
