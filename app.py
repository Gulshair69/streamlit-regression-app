import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def _iter_estimators(obj, seen=None):
    """Recursively yield all nested estimators (Pipeline, GridSearchCV, etc.)."""
    seen = seen or set()
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            yield step
            yield from _iter_estimators(step, seen)
    elif hasattr(obj, "best_estimator_"):
        yield obj.best_estimator_
        yield from _iter_estimators(obj.best_estimator_, seen)
    elif hasattr(obj, "transformers_"):
        for _, trans, _ in obj.transformers_:
            if trans != "drop":
                yield trans
                yield from _iter_estimators(trans, seen)
    elif hasattr(obj, "estimators_") and isinstance(obj.estimators_, (list, tuple)):
        for est in obj.estimators_:
            yield est
            yield from _iter_estimators(est, seen)


def _patch_simple_imputer(model):
    """Fix SimpleImputer _fill_dtype/_fit_dtype for models from older scikit-learn."""
    for obj in _iter_estimators(model):
        if isinstance(obj, SimpleImputer):
            if hasattr(obj, "statistics_"):
                dt = np.result_type(obj.statistics_.dtype, np.float64)
                if not hasattr(obj, "_fill_dtype"):
                    obj._fill_dtype = dt
                if not hasattr(obj, "_fit_dtype"):
                    obj._fit_dtype = dt


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

# Model file options: best_model.pkl + any other .pkl in app folder
_app_dir = Path(__file__).parent if "__file__" in dir() else Path(".")
_pkl_files = list(_app_dir.glob("*.pkl"))
_choices = ["best_model.pkl"] + sorted([f.name for f in _pkl_files if f.name != "best_model.pkl"])
_default_idx = _choices.index("k-NN.pkl") if "k-NN.pkl" in _choices else 0

with st.sidebar:
    model_file = st.selectbox("Model file", _choices, index=_default_idx)
    model = load_model(model_file)
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
        st.info("Tip: Re-run the Heart_Disease_Classification notebook to retrain and save with current scikit-learn.")
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
