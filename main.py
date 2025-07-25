from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import json
import pandas as pd

# Crear la app FastAPI
app = FastAPI()

model_path = "modelos.pkl"
metricas_path = "model_metrics.json"
columns_path = "model_columns.pkl"

# Cargar modelo y recursos
modelo = joblib.load(model_path)

with open(metricas_path, "r", encoding="utf-8") as f:
    metricas = json.load(f)

model_columns = joblib.load(columns_path)

# Definir esquema para entrada
class InputData(BaseModel):
    Age: float
    BMI: float
    Cholesterol: float
    Systolic_BP: float
    Diastolic_BP: float
    Smoking_Status: str
    Alcohol_Intake: float
    Physical_Activity_Level: str
    Family_History: str
    Diabetes: str
    Stress_Level: float
    Salt_Intake: float
    Sleep_Duration: float
    Heart_Rate: float
    LDL: float
    HDL: float
    Triglycerides: float
    Glucose: float
    Gender: str
    Education_Level: str
    Employment_Status: str

@app.get("/")
def read_root():
    return {"msg": "Bienvenido a la API de predicción de hipertensión."}

@app.post("/predecir")
def predecir(data: InputData):
    try:
        d = data.dict()
        df = pd.DataFrame([d])

        df_encoded = pd.get_dummies(df)

        # Reindexar columnas para igualar las usadas en el modelo
        df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

        proba = modelo.predict_proba(df_encoded)[0][1]

        umbral = 0.3
        pred = int(proba >= umbral)

        return {
            "Hypertension_Prediction": "High" if pred == 1 else "Low",
            "Probability": round(proba, 4),
            "Threshold": umbral
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/metricas")
def get_metrics():
    return metricas
