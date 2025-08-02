from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import uvicorn

# Cargar modelo, scaler y columnas
with open("modelo_lasso.pkl", "rb") as f_modelo:
    modelo = pickle.load(f_modelo)

with open("scaler.pkl", "rb") as f_scaler:
    scaler = pickle.load(f_scaler)


# Validar estructura de columnas
def validar_columnas_entrada(df: pd.DataFrame, columnas_entrenadas: list):
    faltantes = set(columnas_entrenadas) - set(df.columns)
    nuevas = set(df.columns) - set(columnas_entrenadas)
    if faltantes or nuevas:
        raise ValueError(
            f"Columnas inconsistentes.\nFaltan: {faltantes}\nSobrantes: {nuevas}"
        )

# Inicializar aplicación FastAPI
app = FastAPI(title="Estimación de Octano NOR con LASSO")

columnas_entrenamiento = [
    '10% Evaporado', # 58.1
    '30% Evaporado', # 72.4
    '50% Evaporado', # 90.2
    '70% Evaporado', # 105.8
    '90% Evaporado', # 130.5
    '95% Evaporado', # 140.1
    'API Gravedad, Digital 60°F', # 33.2
    'Aromáticos', # 25
    'Azufre', # 0.03
    'Benceno', # 0.96
    'Densidad, Digital 15°C' , # 0,733,
    'MTBE', # 20.0%
    'Olefinas', # 12.2
    'Oxígeno total.', # 1.5
    'PIE Evaporado', # 30.2
    'Presión de vapor', # 6.8
    'Punto Final', # 205.0
    'Pérdida', # 1.2
    'TOTAL', # 100.0,
    'PRODUCT_G93NORRM', # 0
    'PRODUCT_G97NORRG', # 1
    'PRODUCT_G97NORRM', # 0
    'PRODUCT_GASOLINA', # 0
    'PRODUCT_REFORMATO', # 0
    'SAMPLE_TYPE_FINAL', # 0
    'SAMPLE_TYPE_POLIDUCTO', # 0
    'SAMPLE_TYPE_FASE_Fase 3', #1
]
# Esquema de entrada con alias que coinciden con columnas originales
class Transaccion(BaseModel):
    evaporado_10: float = Field(..., alias="10% Evaporado")
    evaporado_30: float = Field(..., alias="30% Evaporado")
    evaporado_50: float = Field(..., alias="50% Evaporado")
    evaporado_70: float = Field(..., alias="70% Evaporado")
    evaporado_90: float = Field(..., alias="90% Evaporado")
    evaporado_95: float = Field(..., alias="95% Evaporado")
    api_gravedad: float = Field(..., alias="API Gravedad, Digital 60°F")
    aromaticos: float = Field(..., alias="Aromáticos")
    azufre: float = Field(..., alias="Azufre")
    benceno: float = Field(..., alias="Benceno")
    densidad: float = Field(..., alias="Densidad, Digital 15°C")
    mtbe: float = Field(..., alias="MTBE")
    olefinas: float = Field(..., alias="Olefinas")
    oxigeno_total: float = Field(..., alias="Oxígeno total.")
    pie_evaporado: float = Field(..., alias="PIE Evaporado")
    presion_vapor: float = Field(..., alias="Presión de vapor")
    punto_final: float = Field(..., alias="Punto Final")
    perdida: float = Field(..., alias="Pérdida")
    total: float = Field(..., alias="TOTAL")
    product_g93norrm: float = Field(..., alias="PRODUCT_G93NORRM")
    product_g97norrg: float = Field(..., alias="PRODUCT_G97NORRG")
    product_gasolina: float = Field(..., alias="PRODUCT_GASOLINA")
    product_reformato: float = Field(..., alias="PRODUCT_REFORMATO")
    sample_type_final: float = Field(..., alias="SAMPLE_TYPE_FINAL")
    sample_type_poliducto: float = Field(..., alias="SAMPLE_TYPE_POLIDUCTO")
    sample_type_fase3: float = Field(..., alias="SAMPLE_TYPE_FASE_Fase 3")

# Endpoint para predicción del NOR
@app.get("/")
async def root():
    return {"message": "API para estimación de Octano NOR con LASSO"}


@app.post("/predict_nor")
#"{api_gravedad}_" \
#"{densidad}" \
#"{pie_evaporado}" \
#"{evaporado_10}" \

#)

async def predecir_octano(
    #transaccion: Transaccion,
    api_gravedad: float = 0.0,
    densidad: float = 0.0,
    pie_evaporado: float = 0.0,
    evaporado_10: float = 0.0,
    evaporado_30: float = 0.0,
    evaporado_50: float = 0.0,
    evaporado_70: float = 0.0,
    evaporado_90: float = 0.0,
    evaporado_95: float = 0.0,
    punto_final: float = 0.0,
    pérdida: float = 0.0,
    total: float = 0.0,
    aromaticos: float = 0.0, 
    azufre: float = 0.0,
    benceno: float = 0.0,
    mtbe: float = 0.0, 
    olefinas: float = 0.0,
    oxigeno_total: float = 0.0,  
    presion_vapor: float = 0.0,
    product_g93norrm: float = 0.0,   
    product_g97norrg: float = 0.0,
    product_gasolina: float = 0.0, 
    product_reformato: float = 0.0,
    sample_type_final: float = 0.0,
    sample_type_poliducto: float = 0.0,
    sample_type_fase3: float = 0.0, 
    
):


    try:
        # Convertir entrada a DataFrame y reindexar columnas
        #datos_entrada = pd.DataFrame([transaccion.dict(by_alias=True)])
        #print(datos_entrada)

        # ajuste dataframe [flag: manuel]
        df_aux = pd.DataFrame(
            data={
                "API Gravedad, Digital 60°F": [api_gravedad],
                "Densidad, Digital 15°C": [densidad],
                "PIE Evaporado": [pie_evaporado],
                "10% Evaporado": [evaporado_10],
                "30% Evaporado": [evaporado_30],
                "50% Evaporado": [evaporado_50],
                "70% Evaporado": [evaporado_70],
                "90% Evaporado": [evaporado_90],
                "95% Evaporado": [evaporado_95],
                "Punto Final": [punto_final],
                "Pérdida": [pérdida],
                "TOTAL": [total],
                "Aromáticos": [aromaticos], 
                "Azufre": [azufre],
                "Benceno": [benceno],
                "MTBE": [mtbe], 
                "Olefinas": [olefinas],
                "Oxígeno total.": [oxigeno_total],  
                "Presión de vapor": [presion_vapor],
                "PRODUCT_G93NORRM": [product_g93norrm],   
                "PRODUCT_G97NORRG": [product_g97norrg],
                "PRODUCT_GASOLINA": [product_gasolina], 
                "PRODUCT_REFORMATO": [product_reformato],
                "SAMPLE_TYPE_FINAL": [sample_type_final],
                "SAMPLE_TYPE_POLIDUCTO": [sample_type_poliducto],
                "SAMPLE_TYPE_FASE_Fase 3": [sample_type_fase3]
            }
        )

        datos_entrada = df_aux.reindex(columns=columnas_entrenamiento, fill_value=0)

        # Validar columnas antes de escalar
        validar_columnas_entrada(datos_entrada, columnas_entrenamiento)

        # Escalar datos y predecir
        datos_entrada_scaled = scaler.transform(datos_entrada)
        nor_estimado = modelo.predict(datos_entrada_scaled)[0]

        return {"NOR_estimado": round(nor_estimado, 2)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)



