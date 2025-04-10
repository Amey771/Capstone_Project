import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and feature list
model = joblib.load("xgb_employee_model.pkl")
feature_names = joblib.load("model_features.pkl")

st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")
st.title("🔍 Employee Attrition Predictor")
st.markdown("Fill in the details below to simulate employee attrition prediction:")

# User inputs with sensible defaults
Age = st.slider("Age", 18, 60, 36)
DistanceFromHome = st.slider("Distance From Home (km)", 1, 30, 10)
MonthlyIncome = st.number_input("Monthly Income ($)", 1000, 20000, 5000, step=100)
YearsAtCompany = st.slider("Years at Company", 0, 40, 5)
TotalWorkingYears = st.slider("Total Working Years", 0, 40, 10)
JobLevel = st.selectbox("Job Level", [1, 2, 3, 4, 5], index=1)
OverTime = st.selectbox("OverTime", ["No", "Yes"])
JobSatisfaction = st.slider("Job Satisfaction (1 = Low, 4 = High)", 1, 4, 3)
EnvironmentSatisfaction = st.slider("Environment Satisfaction (1 = Low, 4 = High)", 1, 4, 3)
YearsSinceLastPromotion = st.slider("Years Since Last Promotion", 0, 15, 2)
YearsWithCurrManager = st.slider("Years With Current Manager", 0, 20, 3)
TrainingTimesLastYear = st.slider("Training Times Last Year", 0, 6, 3)

# Create numeric input as a base row
user_input_raw = {
    'Age': Age,
    'DistanceFromHome': DistanceFromHome,
    'MonthlyIncome': MonthlyIncome,
    'YearsAtCompany': YearsAtCompany,
    'TotalWorkingYears': TotalWorkingYears,
    'JobLevel': JobLevel,
    'JobSatisfaction': JobSatisfaction,
    'EnvironmentSatisfaction': EnvironmentSatisfaction,
    'YearsSinceLastPromotion': YearsSinceLastPromotion,
    'YearsWithCurrManager': YearsWithCurrManager,
    'TrainingTimesLastYear': TrainingTimesLastYear,
    'OverTime_Yes': 1 if OverTime == "Yes" else 0
}

# Create a base DataFrame with all model features set to 0
input_df = pd.DataFrame(data=[np.zeros(len(feature_names))], columns=feature_names)

# Fill actual values
for feature, value in user_input_raw.items():
    if feature in input_df.columns:
        input_df.at[0, feature] = value

# Ensure correct dtypes
input_df = input_df.astype(float)

# Predict
if st.button("Predict Attrition Risk"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.markdown("### 🧠 Prediction Result")
    st.success(f"Prediction: {'Attrition (Yes)' if prediction == 1 else 'No Attrition'}")
    st.info(f"Probability of Attrition: **{proba:.2%}**")
