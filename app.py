
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("stroke_rf_model.pkl")

# Define input structure
class PatientData(BaseModel):
    age: int
    gender: int
    hypertension: int
    heart_disease: int
    marital_status: int
    work_type: int
    residence_type: int
    avg_glucose_level: float
    bmi: float
    smoking_status: int
    alcohol_intake: int
    physical_activity: int
    stroke_history: int
    family_history_stroke: int
    dietary_habits: int
    stress_levels: float
    symptoms: int
    systolic_bp: int
    diastolic_bp: int
    hdl: int
    ldl: int

# Preprocess function to return numpy array in correct order
def preprocess(data: PatientData):
    return np.array([
        data.age,
        data.gender,
        data.hypertension,
        data.heart_disease,
        data.marital_status,
        data.work_type,
        data.residence_type,
        data.avg_glucose_level,
        data.bmi,
        data.smoking_status,
        data.alcohol_intake,
        data.physical_activity,
        data.stroke_history,
        data.family_history_stroke,
        data.dietary_habits,
        data.stress_levels,
        data.symptoms,
        data.systolic_bp,
        data.diastolic_bp,
        data.hdl,
        data.ldl
    ]).reshape(1, -1)

@app.get("/")
def root():
    return {"message": "Stroke Prediction API is running"}

@app.post("/predict")
def predict(data: PatientData):
    X = preprocess(data)
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])
    return {"prediction": pred, "probability": proba}

