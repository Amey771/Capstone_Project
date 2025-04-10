
# 🧠 Employee Attrition Predictor – Streamlit App

This interactive web app helps HR professionals and analysts predict whether an employee is at risk of leaving (attrition) based on their profile. The app uses a machine learning model trained with XGBoost and explained using SHAP (SHapley Additive Explanations).

---

## 🚀 Features

- ✅ Predict employee attrition with a tuned XGBoost model
- 🎯 Uses a custom probability threshold for realistic classification
- 📊 Interactive SHAP explainability for transparent predictions
- 🧮 Real-time risk scoring with adjustable profile inputs
- 💡 Clean and simple UI built with Streamlit

---

## 🛠 How to Run Locally

1. **Clone this repo**  
```bash
git clone https://github.com/your-username/employee-attrition-app.git
cd employee-attrition-app
```

2. **Install dependencies**  
```bash
pip install -r requirements.txt
```

3. **Launch the app**  
```bash
streamlit run app.py
```

---

## 🌐 Live Demo

> Coming soon on [Streamlit Cloud](https://share.streamlit.io)

---

## 📁 File Structure

```
📦 employee-attrition-app/
├── app.py                      # Streamlit frontend
├── employee.py                 # Model training and export
├── xgb_employee_model.pkl      # Trained XGBoost model
├── final_model_features.pkl    # Feature names used by the model
├── final_default_values.pkl    # Default values used for prediction UI
├── dropdown_options.pkl        # Categorical dropdown values
├── requirements.txt            # Dependencies
└── README.md                   # You're reading this!
```

---

## 📊 Model Details

- **Algorithm**: XGBoost Classifier
- **Tuning**: RandomizedSearchCV with ROC AUC scoring
- **Handling Imbalance**: `scale_pos_weight`
- **Explainability**: SHAP TreeExplainer for bar/force plots

---

## 🤝 Contributors

👤 Amey Suresh Borkar  
📧 amey.borkar01@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/ameyborkar771) | [GitHub](https://github.com/Amey771)

---

## 📄 License

MIT License — use it, share it, build on it.
