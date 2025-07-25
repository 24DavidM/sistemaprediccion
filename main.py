from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import pandas as pd
import requests
import os
import gdown

# Crear la app FastAPI
app = FastAPI()

# URLs de descarga directa desde Google Drive
MODEL_URL = "https://drive.google.com/uc?export=download&id=1T2MsernqFCNKmfpGmb2cTyn1w6zKRK_2"
COLUMNS_URL = "https://drive.google.com/uc?export=download&id=1o3DWpvV89wA7oPp0PzOy8iWoFsyXqw_o"
METRICAS_FILE = "model_metrics.json"

# Archivos locales
MODEL_FILE = "modelos.pkl"
COLUMNS_FILE = "model_columns.pkl"

# Función para descargar usando gdown (mejor manejo para Drive)
def download_file(url: str, filename: str):
    if not os.path.exists(filename):
        print(f"Descargando {filename} desde {url} ...")
        gdown.download(url, filename, quiet=False)
        print(f"{filename} descargado correctamente.")
    else:
        print(f"{filename} ya existe, no se descarga.")

# Descargar los modelos antes de cargarlos
download_file(MODEL_URL, MODEL_FILE)
download_file(COLUMNS_URL, COLUMNS_FILE)

# Cargar modelo y recursos locales
modelo = joblib.load(MODEL_FILE)
model_columns = joblib.load(COLUMNS_FILE)

with open(METRICAS_FILE, "r", encoding="utf-8") as f:
    metricas = json.load(f)

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
