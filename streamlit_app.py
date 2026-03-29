import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Diabetic Retinopathy Predictor", page_icon="👁️", layout="centered")

BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    BASE_DIR / "retinopathy_model.pkl",
    Path.cwd() / "retinopathy_model.pkl",
    BASE_DIR / "src" / "retinopathy_model.pkl",
    Path.cwd() / "src" / "retinopathy_model.pkl",
]
FEATURES = ["age", "systolic_bp", "diastolic_bp", "cholesterol"]


def load_model(model_path: Path):
    with model_path.open("rb") as f:
        return pickle.load(f)


def resolve_model_path() -> Optional[Path]:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    return None


def predict_retinopathy(model, values: dict):
    row = pd.DataFrame([values], columns=FEATURES)
    pred = model.predict(row)[0]

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(row)[0][1])

    return int(pred), probability


st.title("Diabetic Retinopathy Prediction")
st.write("Enter clinical values to predict if retinopathy is likely.")

model_path = resolve_model_path()
if model_path is None:
    searched_paths = "\n".join(str(p) for p in MODEL_CANDIDATES)
    st.error(f"Model file not found. Searched:\n{searched_paths}")
    st.stop()

model = load_model(model_path)


def parse_float_input(label: str, raw_value: str):
    if raw_value is None or raw_value.strip() == "":
        return None, f"Please enter {label}."

    try:
        return float(raw_value.strip()), None
    except ValueError:
        return None, f"{label} must be a valid number."


col1, col2 = st.columns(2)
with col1:
    age_raw = st.text_input("Age", value="", placeholder="Enter age")
    systolic_bp_raw = st.text_input("Systolic BP", value="", placeholder="Enter systolic BP")
with col2:
    diastolic_bp_raw = st.text_input("Diastolic BP", value="", placeholder="Enter diastolic BP")
    cholesterol_raw = st.text_input("Cholesterol", value="", placeholder="Enter cholesterol")

if st.button("Predict", type="primary"):
    age, age_err = parse_float_input("Age", age_raw)
    systolic_bp, systolic_bp_err = parse_float_input("Systolic BP", systolic_bp_raw)
    diastolic_bp, diastolic_bp_err = parse_float_input("Diastolic BP", diastolic_bp_raw)
    cholesterol, cholesterol_err = parse_float_input("Cholesterol", cholesterol_raw)

    errors = [e for e in [age_err, systolic_bp_err, diastolic_bp_err, cholesterol_err] if e]
    if errors:
        for err in errors:
            st.warning(err)
        st.stop()

    payload = {
        "age": age,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "cholesterol": cholesterol,
    }

    pred, proba = predict_retinopathy(model, payload)

    if pred == 1:
        st.error("Prediction: Retinopathy likely")
    else:
        st.success("Prediction: No retinopathy likely")

    if proba is not None:
        st.metric("Probability of Retinopathy", f"{proba * 100:.2f}%")

st.caption(f"Model: XGBoost (loaded from {model_path})")
