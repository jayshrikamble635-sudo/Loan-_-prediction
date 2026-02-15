import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("loan_prediction_model.pkl")
encoder = joblib.load("loan_encoder.pkl")

st.title("Loan Prediction App")

# Inputs
gender = st.selectbox("Gender", encoder["Gender"].classes_)
education = st.selectbox("Education", encoder["Education"].classes_)
income = st.number_input("Applicant Income", min_value=0)

# Create DataFrame
df = pd.DataFrame({
    "Gender": [gender],
    "Education": [education],
    "ApplicantIncome": [income]
})

# Prediction
if st.button("Predict"):
    df["Gender"] = encoder["Gender"].transform(df["Gender"])
    df["Education"] = encoder["Education"].transform(df["Education"])

    prediction = model.predict(df)

    if prediction[0] == 1:
        st.write("Loan Approved")
    else:
        st.write("Loan Rejected")
