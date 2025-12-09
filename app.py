
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model + encoders
model = joblib.load("stroke_rf_model.pkl")
encoders = joblib.load("encoders.pkl")

app = FastAPI()

class PatientData(BaseModel):
    age: int
    gender: str
    hypertension: int
    heart_disease: int
    marital_status: str
    work_type: str
    residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str
    alcohol_intake: str
    physical_activity: str
    stroke_history: int
    family_history_stroke: str
    dietary_habits: str
    stress_levels: float
    symptoms: int
    systolic_bp: int
    diastolic_bp: int
    hdl: int
    ldl: int

# ✔️ Apply encoders EXACTLY as in training
def preprocess(data: PatientData):

    # Apply label encoders
    gender = encoders["Gender"].transform([data.gender])[0]
    marital = encoders["Marital Status"].transform([data.marital_status])[0]
    work = encoders["Work Type"].transform([data.work_type])[0]
    residence = encoders["Residence Type"].transform([data.residence_type])[0]
    smoking = encoders["Smoking Status"].transform([data.smoking_status])[0]
    alcohol = encoders["Alcohol Intake"].transform([data.alcohol_intake])[0]
    physical = encoders["Physical Activity"].transform([data.physical_activity])[0]
    family = encoders["Family History of Stroke"].transform([data.family_history_stroke])[0]
    diet = encoders["Dietary Habits"].transform([data.dietary_habits])[0]

    # ✔️ Keep numeric fields as they are
    arr = np.array([
        data.age,
        gender,
        data.hypertension,
        data.heart_disease,
        marital,
        work,
        residence,
        data.avg_glucose_level,
        data.bmi,
        smoking,
        alcohol,
        physical,
        data.stroke_history,
        family,
        diet,
        data.stress_levels,
        data.symptoms,
        data.systolic_bp,
        data.diastolic_bp,
        data.hdl,
        data.ldl
    ]).reshape(1, -1)

    return arr

@app.get("/")
def root():
    return {"message": "Stroke Prediction API is running"}

@app.post("/predict")
def predict(data: PatientData):
    X = preprocess(data)
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    return {"prediction": prediction, "probability": probability}


