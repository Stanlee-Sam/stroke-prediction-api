
from fastapi import FastAPI, HTTPException
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

def transform_with_encoder(encoder, value, col_name):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        # handle unseen label
        return -1  # or some default/fallback integer

def preprocess(data: PatientData):
    # Normalize strings
    gender_val = data.gender.strip().title()
    marital_val = data.marital_status.strip().title()
    work_val = data.work_type.strip().title()
    residence_val = data.residence_type.strip().title()
    smoking_val = data.smoking_status.strip().lower()  # match encoder
    alcohol_val = data.alcohol_intake.strip().title()
    physical_val = data.physical_activity.strip().title()
    family_val = data.family_history_stroke.strip().title()
    diet_val = data.dietary_habits.strip().title()

    # Apply encoders safely
    gender = transform_with_encoder(encoders["Gender"], gender_val, "Gender")
    marital = transform_with_encoder(encoders["Marital Status"], marital_val, "Marital Status")
    work = transform_with_encoder(encoders["Work Type"], work_val, "Work Type")
    residence = transform_with_encoder(encoders["Residence Type"], residence_val, "Residence Type")
    smoking = transform_with_encoder(encoders["Smoking Status"], smoking_val, "Smoking Status")
    alcohol = transform_with_encoder(encoders["Alcohol Intake"], alcohol_val, "Alcohol Intake")
    physical = transform_with_encoder(encoders["Physical Activity"], physical_val, "Physical Activity")
    family = transform_with_encoder(encoders["Family History of Stroke"], family_val, "Family History of Stroke")
    diet = transform_with_encoder(encoders["Dietary Habits"], diet_val, "Dietary Habits")

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
    try:
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"prediction": prediction, "probability": probability}


