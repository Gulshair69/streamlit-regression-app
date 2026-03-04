import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def _patch_simple_imputer(obj):
    """Fix SimpleImputer _fill_dtype compatibility for models trained with older scikit-learn."""
    if isinstance(obj, SimpleImputer):
        if not hasattr(obj, "_fill_dtype") and hasattr(obj, "statistics_"):
            obj._fill_dtype = np.result_type(obj.statistics_.dtype, np.float64)
    elif isinstance(obj, Pipeline):
        for _, step in obj.steps:
            _patch_simple_imputer(step)


@st.cache_resource
def load_model(model_path: str = "best_model.pkl"):
    path = Path(model_path)
    if not path.exists():
        st.error(f"Model file '{model_path}' not found. Run the Heart_Disease_Classification notebook to train and save it.")
        return None
    model = joblib.load(path)
    _patch_simple_imputer(model)
    return model


st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered")

st.title("Heart Disease Risk Predictor")
st.markdown(
    """
This app uses a machine learning model trained on the classic heart disease dataset
(`heart.csv`) to estimate the **risk of heart disease** for a given patient profile.
"""
)

model = load_model()

with st.sidebar:
    st.header("Patient Features")

    age = st.slider("Age", min_value=20, max_value=90, value=50)
    sex = st.selectbox("Sex (1 = male, 0 = female)", options=[0, 1], index=1)
    cp = st.selectbox(
        "Chest pain type (cp)",
        options=[0, 1, 2, 3],
        format_func=lambda x: f"{x}",
    )
    trestbps = st.slider("Resting blood pressure (trestbps)", 80, 200, 130)
    chol = st.slider("Serum cholesterol (chol)", 100, 600, 240)
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", options=[0, 1])
    restecg = st.selectbox("Resting ECG results (restecg)", options=[0, 1, 2])
    thalach = st.slider("Max heart rate achieved (thalach)", 70, 220, 150)
    exang = st.selectbox("Exercise induced angina (exang)", options=[0, 1])
    oldpeak = st.slider("ST depression induced by exercise (oldpeak)", 0.0, 6.5, 1.0, 0.1)
    slope = st.selectbox("Slope of peak exercise ST segment (slope)", options=[0, 1, 2])
    ca = st.selectbox("Number of major vessels (ca)", options=[0, 1, 2, 3, 4], index=0)
    thal = st.selectbox("Thal (0 = unknown, 1 = normal, 2 = fixed defect, 3 = reversable)", options=[0, 1, 2, 3], index=3)

    predict_btn = st.button("Predict Risk")


if model is not None and predict_btn:
    input_data = pd.DataFrame(
        [
            {
                "age": age,
                "sex": sex,
                "cp": cp,
                "trestbps": trestbps,
                "chol": chol,
                "fbs": fbs,
                "restecg": restecg,
                "thalach": thalach,
                "exang": exang,
                "oldpeak": oldpeak,
                "slope": slope,
                "ca": ca,
                "thal": thal,
            }
        ]
    )

    try:
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[:, 1][0]
        pred = int(model.predict(input_data)[0])
    except Exception as e:
        st.error(f"Error while making prediction: {e}")
        st.stop()

    st.subheader("Prediction")
    if proba is not None:
        st.write(f"Estimated probability of heart disease: **{proba:.2%}**")

    if pred == 1:
        st.warning("The model predicts **heart disease present (class 1)** for this profile.")
    else:
        st.success("The model predicts **no heart disease (class 0)** for this profile.")

    st.caption(
        "This tool is for educational purposes only and must not be used as a substitute for professional medical advice."
    )
