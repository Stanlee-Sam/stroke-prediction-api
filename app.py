
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
    hypertension: int
    heart_disease: int
    avg_glucose_level: float
    bmi: float
    stress_levels: float

# Preprocessing (same as training)
def preprocess(data):
    arr = np.array([
        data.age,
        data.hypertension,
        data.heart_disease,
        data.avg_glucose_level,
        data.bmi,
        data.stress_levels
    ]).reshape(1, -1)
    return arr

@app.get("/")
def root():
    return {"message": "Stroke Prediction API is running"}

@app.post("/predict")
def predict(data: PatientData):
    X = preprocess(data)
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])
    return {"prediction": pred, "probability": proba}


