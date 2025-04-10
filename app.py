import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load assets
model = joblib.load("xgb_employee_model.pkl")
feature_names = joblib.load("final_model_features.pkl")
default_values = joblib.load("final_default_values.pkl")
dropdowns = joblib.load("dropdown_options.pkl")

# App UI
st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")
st.title("🔍 Employee Attrition Predictor")
st.markdown("Provide employee details to estimate attrition risk.")

# Numeric & categorical input widgets
Age = st.slider("Age", 18, 60, int(default_values.get("Age", 36)))
DistanceFromHome = st.slider("Distance From Home (km)", 1, 30, int(default_values.get("DistanceFromHome", 7)))
MonthlyIncome = st.number_input("Monthly Income ($)", 1000, 20000, int(default_values.get("MonthlyIncome", 5000)), step=100)
YearsAtCompany = st.slider("Years at Company", 0, 40, int(default_values.get("YearsAtCompany", 5)))
TotalWorkingYears = st.slider("Total Working Years", 0, 40, int(default_values.get("TotalWorkingYears", 10)))
YearsSinceLastPromotion = st.slider("Years Since Last Promotion", 0, 15, int(default_values.get("YearsSinceLastPromotion", 2)))
YearsWithCurrManager = st.slider("Years With Current Manager", 0, 20, int(default_values.get("YearsWithCurrManager", 3)))
TrainingTimesLastYear = st.slider("Training Times Last Year", 0, 6, int(default_values.get("TrainingTimesLastYear", 3)))
PercentSalaryHike = st.slider("Percent Salary Hike", 10, 25, int(default_values.get("PercentSalaryHike", 15)))
StockOptionLevel = st.selectbox("Stock Option Level", [0, 1, 2, 3], index=int(default_values.get("StockOptionLevel", 1)))
JobLevel = st.selectbox("Job Level", [1, 2, 3, 4, 5], index=int(default_values.get("JobLevel", 2) - 1))
JobInvolvement = st.slider("Job Involvement (1–4)", 1, 4, int(default_values.get("JobInvolvement", 3)))
JobSatisfaction = st.slider("Job Satisfaction (1–4)", 1, 4, int(default_values.get("JobSatisfaction", 3)))
EnvironmentSatisfaction = st.slider("Environment Satisfaction (1–4)", 1, 4, int(default_values.get("EnvironmentSatisfaction", 3)))
WorkLifeBalance = st.slider("Work Life Balance (1–4)", 1, 4, int(default_values.get("WorkLifeBalance", 3)))
OverTime = st.selectbox("OverTime", ["No", "Yes"], index=int(default_values.get("OverTime_Yes", 0)))

# Dropdowns (categorical features)
BusinessTravel = st.selectbox("Business Travel", dropdowns["BusinessTravel"])
Department = st.selectbox("Department", dropdowns["Department"])
JobRole = st.selectbox("Job Role", dropdowns["JobRole"])
MaritalStatus = st.selectbox("Marital Status", dropdowns["MaritalStatus"])
EducationField = st.selectbox("Education Field", dropdowns["EducationField"])
Gender = st.selectbox("Gender", dropdowns["Gender"])

# Base DataFrame
input_data = pd.DataFrame([default_values])
input_data = input_data[feature_names]  # ensure correct column order

# Overwrite with user input
input_data.at[0, "Age"] = Age
input_data.at[0, "DistanceFromHome"] = DistanceFromHome
input_data.at[0, "MonthlyIncome"] = MonthlyIncome
input_data.at[0, "YearsAtCompany"] = YearsAtCompany
input_data.at[0, "TotalWorkingYears"] = TotalWorkingYears
input_data.at[0, "YearsSinceLastPromotion"] = YearsSinceLastPromotion
input_data.at[0, "YearsWithCurrManager"] = YearsWithCurrManager
input_data.at[0, "TrainingTimesLastYear"] = TrainingTimesLastYear
input_data.at[0, "PercentSalaryHike"] = PercentSalaryHike
input_data.at[0, "StockOptionLevel"] = StockOptionLevel
input_data.at[0, "JobLevel"] = JobLevel
input_data.at[0, "JobInvolvement"] = JobInvolvement
input_data.at[0, "JobSatisfaction"] = JobSatisfaction
input_data.at[0, "EnvironmentSatisfaction"] = EnvironmentSatisfaction
input_data.at[0, "WorkLifeBalance"] = WorkLifeBalance
input_data.at[0, "OverTime_Yes"] = 1 if OverTime == "Yes" else 0

# One-hot encode categorical fields
for prefix, value in {
    "BusinessTravel": BusinessTravel,
    "Department": Department,
    "JobRole": JobRole,
    "MaritalStatus": MaritalStatus,
    "EducationField": EducationField,
    "Gender": Gender,
}.items():
    col_name = f"{prefix}_{value}"
    if col_name in input_data.columns:
        input_data.at[0, col_name] = 1

# Ensure proper types
input_data = input_data.astype(float)

st.write("🧾 Input Summary", input_data.T)

col_name = f"JobRole_{JobRole}"
st.write("🧪 Checking:", col_name in input_data.columns, input_data.at[0, col_name] if col_name in input_data.columns else "Missing")


risky_flags = [
    "OverTime_Yes",
    "JobRole_Sales Representative",
    "BusinessTravel_Travel_Frequently",
    "WorkLifeBalance",
    "JobSatisfaction"
]

st.subheader("🔍 Risk Feature Check:")
for col in risky_flags:
    st.write(f"{col}: ", input_data.at[0, col] if col in input_data.columns else "❌ MISSING")


# # Predict
# if st.button("Predict Attrition Risk"):
#     prediction = model.predict(input_data)[0]
#     proba = model.predict_proba(input_data)[0][1]

#     st.markdown("### 🧠 Prediction Result")
#     st.success(f"Prediction: {'Attrition (Yes)' if prediction else 'No Attrition'}")
#     st.info(f"Probability of Attrition: **{proba:.2%}**")


# # Predict
# if st.button("Predict Attrition Risk"):
#     prediction_proba = model.predict_proba(input_data)[0][1]

#     # threshold = st.slider("Custom Risk Threshold (%)", 10, 90, 35) / 100

#     threshold = 30

#     prediction_label = "Attrition Risk" if prediction_proba >= threshold else "No Attrition"

#     st.markdown("### 🧠 Prediction Result")
#     if prediction_label == "Attrition Risk":
#         st.error(f"Prediction: {prediction_label}")
#     else:
#         st.success(f"Prediction: {prediction_label}")

#     st.info(f"Probability of Attrition: **{prediction_proba:.2%}** (Threshold: {threshold*100:.0f}%)")


st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #2e8b57;
        color: white;
        border-radius: 8px;
        padding: 0.6em 1.5em;
        font-weight: 600;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)


# --- Threshold & Prediction Logic ---
 
# make this a slider
# threshold = st.slider("Custom Risk Threshold (%)", 10, 90, 35) / 100

threshold = 0.35  # 35%
if st.button("Predict Attrition Risk"):
    proba = model.predict_proba(input_data)[0][1]  # this is 0.0059, or 0.59%
    percent_proba = proba * 100
    is_at_risk = percent_proba >= threshold

    st.markdown("### 🧠 Prediction Result")
    if is_at_risk:
        st.error("🔴 Prediction: **Attrition Risk**")
    else:
        st.success("🟢 Prediction: **No Risk of Attrition**")

    st.caption(f"🧮 Risk Score: **{percent_proba:.2%}** (Threshold: {threshold * 100:.1f}%)")





import shap
import matplotlib.pyplot as plt

# SHAP Explainer Setup
explainer = shap.TreeExplainer(model)

# Ensure input format and type
shap_input = input_data.astype(float)

# Compute SHAP values
shap_values = explainer.shap_values(shap_input)

# Display SHAP force plot or bar chart
st.subheader("🔎 Feature Impact (SHAP Explanation)")

# Bar plot (simpler for Streamlit)
shap.summary_plot(shap_values, shap_input, plot_type="bar", show=False)
st.pyplot(plt.gcf())


# # Force plot for individual prediction
# st.set_option('deprecation.showPyplotGlobalUse', False)
# shap.force_plot(explainer.expected_value, shap_values[0], shap_input, matplotlib=True, show=False)
# st.pyplot()
