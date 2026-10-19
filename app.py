import streamlit as st
import pandas as pd
import joblib

# ----------------------------------
# 1. Page Configuration
# ----------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ----------------------------------
# 2. Load the Trained Model
# ----------------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/customer_churn_model.pkl")


model = load_model()

# ----------------------------------
# 3. Application Title
# ----------------------------------

st.title("📊 Customer Churn Prediction System")

st.write(
    "A Machine Learning application that estimates "
    "whether a telecom customer is likely to leave."
)

st.info("Enter the customer's information below.")

# ----------------------------------
# 4. Customer Information
# ----------------------------------

st.header("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender", ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen", [0, 1]
    )

    partner = st.selectbox(
        "Has Partner?", ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Has Dependents?", ["Yes", "No"]
    )

    tenure = st.slider(
        "Tenure (months)", 0, 72, 12
    )

    phone_service = st.selectbox(
        "Phone Service", ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

with col2:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

with col3:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing", ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )

    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=100000.0,
        value=840.0,
        step=10.0
    )

# ----------------------------------
# 5. Prepare Customer Data
# ----------------------------------

customer_data = {
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

# ----------------------------------
# 6. Predict Customer Churn
# ----------------------------------

if st.button("Predict Customer Churn", type="primary"):

    customer_df = pd.DataFrame([customer_data])

    churn_probability = model.predict_proba(
        customer_df
    )[0][1]

    threshold = 0.55

    st.header("Prediction Results")

    st.metric(
        "Estimated Churn Probability",
        f"{churn_probability:.2%}"
    )

    st.progress(float(churn_probability))

    if churn_probability >= threshold:
        st.error("⚠️ Prediction: Likely to Churn")

        st.write(
            "Consider reviewing this customer's service "
            "experience and support needs."
        )

    else:
        st.success("✅ Prediction: Likely to Stay")

        st.write(
            "The model estimates a lower likelihood of churn."
        )

    st.caption(
        "This is a machine-learning estimate, not a guarantee. "
        "The 55% decision threshold is experimental and "
        "requires validation before real-world use."
    )