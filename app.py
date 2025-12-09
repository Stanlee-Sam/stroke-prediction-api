
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

   # Normalize strings
    gender_val = data.gender.strip().title()  # 'male' -> 'Male'
    marital_val = data.marital_status.strip().title()
    work_val = data.work_type.strip().title()
    residence_val = data.residence_type.strip().title()
    smoking_val = data.smoking_status.strip().lower()  # match encoder
    alcohol_val = data.alcohol_intake.strip().title()
    physical_val = data.physical_activity.strip().title()
    family_val = data.family_history_stroke.strip().title()
    diet_val = data.dietary_habits.strip().title()
    
    # Apply label encoders
    gender = encoders["Gender"].transform([gender_val])[0]
    marital = encoders["Marital Status"].transform([marital_val])[0]
    work = encoders["Work Type"].transform([work_val])[0]
    residence = encoders["Residence Type"].transform([residence_val])[0]
    smoking = encoders["Smoking Status"].transform([smoking_val])[0]
    alcohol = encoders["Alcohol Intake"].transform([alcohol_val])[0]
    physical = encoders["Physical Activity"].transform([physical_val])[0]
    family = encoders["Family History of Stroke"].transform([family_val])[0]
    diet = encoders["Dietary Habits"].transform([diet_val])[0]

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


