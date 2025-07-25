from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import pandas as pd

# Crear la app FastAPI
app = FastAPI()

MODEL_URL = "https://drive.google.com/uc?export=download&id=1eLWS_EmFjMgp1zkqz6Xg3U3X0JBcH-P4"
metricas_path = "model_metrics.json"
COLUMNS_URL = "https://drive.google.com/uc?export=download&id=1j6m_u1-rBTIjuXNU0BVNpTzd1IQQJk6z"

# Cargar modelo y recursos
modelo = joblib.load(MODEL_URL)

with open(metricas_path, "r", encoding="utf-8") as f:
    metricas = json.load(f)

model_columns = joblib.load(COLUMNS_URL)

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
