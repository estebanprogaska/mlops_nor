import pickle
import numpy as np
import pandas as pd

import pickle

# Cargar el modelo y el scaler
with open('modelo_lasso.pkl', 'rb') as archivo_modelo:
    modelo = pickle.load(archivo_modelo)

with open('scaler.pkl', 'rb') as archivo_scaler:
    scaler = pickle.load(archivo_scaler)

# with open("columnas_entrenamiento.pkl", "wb") as f:
#     pickle.dump(columnas_entrenamiento, f)

columnas = [
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

# Recolectar valores del usuario
valores_usuario = []

print("Ingreso de variables para predicción NOR:\n")
for columna in columnas:
    while True:
        try:
            valor = float(input(f"{columna}: "))
            valores_usuario.append(valor)
            break
        except ValueError:
            print("Entrada inválida. Por favor, ingresa un número.")

# Convertir a DataFrame
nueva_muestra = pd.DataFrame([valores_usuario], columns=columnas)

# Escalar características
nueva_muestra_scaled = scaler.transform(nueva_muestra)

# Realizar la predicción
prediccion_nor = modelo.predict(nueva_muestra_scaled)[0]

# Mostrar el resultado
print(f"\nNúmero de Octano Research estimado: {prediccion_nor:.2f}")