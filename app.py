import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- Model Loading ---
# Note: This app requires 'best_rf_model.joblib' and 'scaler.joblib'
# to be present in the same directory when running locally with Streamlit.
try:
    model = joblib.load("best_rf_model.joblib")
    scaler = joblib.load("scaler.joblib")
    model_loaded = True
except FileNotFoundError:
    st.error("🚨 Error: Model files ('best_rf_model.joblib' and/or 'scaler.joblib') not found.")
    st.info("Please ensure your trained model and scaler files are in the same directory.")
    model_loaded = False
except Exception as e:
    st.error(f"🚨 Error loading model or scaler: {e}")
    model_loaded = False


st.set_page_config(page_title="Medical Condition Prediction", layout="centered")
st.title("🩺 Medical Condition Prediction System")
st.markdown("Enter patient physiological and lifestyle metrics below. This model uses these inputs to predict a potential medical condition, reflecting how different factors contribute to overall health. [Image of Human Physiological Risk Factors and Disease]")

# --- Input Fields ---
st.header("Patient Metrics")

# Input fields aligned with typical features in health prediction models
age = st.number_input("Age (Years)", min_value=0, max_value=120, value=30, help="Patient's age.")
glucose = st.number_input("Glucose (mg/dL)", min_value=50.0, max_value=300.0, value=100.0, help="Fasting blood glucose level.")
bp = st.number_input("Blood Pressure (mmHg)", min_value=60.0, max_value=200.0, value=120.0, help="Systolic blood pressure.")
bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=25.0, help="Body Mass Index.")
cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100.0, max_value=400.0, value=180.0, help="Total Cholesterol level.")
hba1c = st.number_input("HbA1c (%)", min_value=4.0, max_value=14.0, value=5.5, help="Glycated Hemoglobin level, measures average blood sugar over 3 months.")
triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=50.0, max_value=500.0, value=150.0, help="Blood triglyceride level.")

st.header("Lifestyle Factors")
diet_score = st.number_input("Diet Score (0=Poor, 10=Excellent)", min_value=0, max_value=10, value=5, help="Subjective score reflecting diet quality.")
stress = st.number_input("Stress Level (0=Low, 10=High)", min_value=0, max_value=10, value=5, help="Subjective stress level.")
sleep = st.number_input("Sleep Hours (Hours)", min_value=0.0, max_value=24.0, value=7.0, help="Average hours of sleep per night.")

# --- Prediction Logic ---
if st.button("Predict Medical Condition", disabled=not model_loaded):
    if model_loaded:
        # Convert input into DataFrame
        input_df = pd.DataFrame([[
            age, glucose, bp, bmi, cholesterol,
            hba1c, triglycerides, diet_score, stress, sleep
        ]], columns=[
            "Age", "Glucose", "Blood Pressure", "BMI",
            "Cholesterol", "HbA1c", 'Triglycerides', "Diet Score", "Stress Level", "Sleep Hours"
        ])

        try:
            # Scale and predict
            scaled_input = scaler.transform(input_df)
            prediction_class = model.predict(scaled_input)[0]
            # -----------------------------------------------------
            # Map numeric prediction back to readable labels
            condition_labels = {
                0: "Diabetes",
                1: "Healthy",
                2: "Asthma",
                3: "Obesity",
                4: "Hypertension",
                5: "Arthritis",
                6: "Cancer"
            }

            result = condition_labels.get(prediction_class, f"Class {prediction_class} (Unknown)")
            
            # Display results with styling
            st.success(f"### ✅ Predicted Medical Condition: **{result}**")
            
       

        except Exception as e:
            st.error(f"⚠️ Prediction Error: Could not process inputs. Details: {e}")
    else:
        st.warning("Model cannot run without the required joblib files.")

st.write("---")
st.caption("Developed by Sebin Mathew | Deployed with Streamlit 🚀")