import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import os

# --- DEFINICIÓN DE MODELOS DE DATOS ---
class EmbarazoRequest(BaseModel):
    M14: float  # Controles
    V501: float # Estado Civil
    V714: float # Trabajo
    V701: float # Educacion Pareja
    M13: float  # Mes primer control

class ViolenciaRequest(BaseModel):
    V501: float
    V701: float
    V714: float
    V743A: float
    D113: float

# --- INICIO DE LA APP ---
app = FastAPI(title="API Predicción ENDES", version="1.0")

# Variables Globales
spark = None
model_embarazo = None
model_violencia = None

@app.on_event("startup")
def load_models():
    global spark, model_embarazo, model_violencia
    
    print(">>> [API] Iniciando Spark Session...")
    # Nota: No necesitamos configurar master ni memoria aquí, 
    # porque spark-submit ya se encarga de eso.
    spark = SparkSession.builder \
        .appName("ENDES_API_Server") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    print(">>> [API] Cargando Modelos...")
    
    # Cargar Modelo Embarazo
    if os.path.exists("data/processed/modelo_RIESGO_EMBARAZO"):
        try:
            model_embarazo = PipelineModel.load("data/processed/modelo_RIESGO_EMBARAZO")
            print("   -> Modelo Embarazo: CARGADO EXITOSAMENTE")
        except Exception as e:
            print(f"   -> Error Embarazo: {e}")
    else:
        print("   -> Modelo Embarazo: NO ENCONTRADO (Verifica la carpeta data/processed)")

    # Cargar Modelo Violencia
    if os.path.exists("data/processed/modelo_VIOLENCIA_FAMILIAR"):
        try:
            model_violencia = PipelineModel.load("data/processed/modelo_VIOLENCIA_FAMILIAR")
            print("   -> Modelo Violencia: CARGADO EXITOSAMENTE")
        except Exception as e:
            print(f"   -> Error Violencia: {e}")
    else:
        print("   -> Modelo Violencia: NO ENCONTRADO")

@app.get("/")
def home():
    return {"estado": "online", "mensaje": "API corriendo sobre Spark. Usa /docs para probar."}

def rellenar_columnas(df, modelo):
    """Rellena columnas faltantes con 0 para evitar errores en transform()"""
    if not modelo: return df
    try:
        cols_esperadas = modelo.stages[0].getInputCols()
        for c in cols_esperadas:
            if c not in df.columns:
                df = df.withColumn(c, df[df.columns[0]] * 0)
    except: pass
    return df

@app.post("/predecir/embarazo")
def predict_embarazo(data: EmbarazoRequest):
    if not model_embarazo:
        raise HTTPException(status_code=500, detail="Modelo Embarazo no cargado")
    
    try:
        df = spark.createDataFrame([data.dict()])
        df = rellenar_columnas(df, model_embarazo)
        
        res = model_embarazo.transform(df).select("prediction", "probability").first()
        
        return {
            "riesgo": int(res["prediction"]),
            "probabilidad": round(float(res["probability"][1]), 4),
            "alerta": "SI" if res["prediction"] == 1 else "NO"
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/predecir/violencia")
def predict_violencia(data: ViolenciaRequest):
    if not model_violencia:
        raise HTTPException(status_code=500, detail="Modelo Violencia no cargado")
    
    try:
        df = spark.createDataFrame([data.dict()])
        df = rellenar_columnas(df, model_violencia)
        
        res = model_violencia.transform(df).select("prediction", "probability").first()
        
        return {
            "violencia": int(res["prediction"]),
            "probabilidad": round(float(res["probability"][1]), 4),
            "alerta": "SI" if res["prediction"] == 1 else "NO"
        }
    except Exception as e:
        return {"error": str(e)}

# --- BLOQUE CLAVE: EJECUTAR SERVIDOR DESDE DENTRO ---
if __name__ == "__main__":
    print(">>> [LANZADOR] Iniciando servidor Uvicorn a través de Spark-Submit...")
    # Esto lanza el servidor web en el puerto 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)