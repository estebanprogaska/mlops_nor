# nor_lasso_full.py

import pickle
import sklearn
from packaging import version
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Estilo visual
sns.set_theme(style="whitegrid")
COLOR = "#32BAD9"

# Cargar datos
df = pd.read_csv("04_eda_clean.csv")
componentes = pd.read_csv("03_componentes.csv")  # No se usa explícitamente pero se incluye

# Preprocesamiento
df.drop("ID_NUMERIC", axis=1, errors="ignore", inplace=True)
df["PRODUCT"] = df["PRODUCT"].astype("category")
df["SAMPLE_TYPE"] = df["SAMPLE_TYPE"].astype("category")
df["SAMPLE_TYPE_FASE"] = df["SAMPLE_TYPE_FASE"].astype("category")

# División X / y
X = df.drop(columns=["Número de Octano Research"])
X = pd.get_dummies(X, columns=["PRODUCT", "SAMPLE_TYPE", "SAMPLE_TYPE_FASE"], drop_first=True)
y = df["Número de Octano Research"]

# Escalamiento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split de entrenamiento / test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Nombre las variables para el modelo
X_train = pd.DataFrame(X_train, columns=X.columns)
print(f"Variables de entrada: {X_train.columns.tolist()}")

# Validación cruzada para elegir alpha
lasso_cv = LassoCV(alphas=np.logspace(-5, 5, 1000), cv=5, max_iter=10000)
lasso_cv.fit(X_train, y_train)
mejor_alpha = lasso_cv.alpha_
print(f"\n✅ Mejor alpha encontrado: {mejor_alpha:.6f}")

# Entrenamiento del modelo final
modelo_final = Lasso(alpha=mejor_alpha, max_iter=10000)
modelo_final.fit(X_train, y_train)

# Predicciones
y_pred = modelo_final.predict(X_test)

# Métricas
mae = mean_absolute_error(y_test, y_pred)
#rmse = mean_squared_error(y_test, y_pred, squared=False)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("\n📊 Métricas del modelo LASSO:")
print(f"MAE:   {mae:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"R2:    {r2:.4f}")
print(f"MAPE:  {mape:.4f} %")


# Guardar el modelo LASSO en un archivo .pkl
with open('modelo_lasso.pkl', 'wb') as archivo_salida:
    pickle.dump(modelo_final, archivo_salida)


# Si también deseas guardar el scaler
with open('scaler.pkl', 'wb') as archivo_salida:
    pickle.dump(scaler, archivo_salida)

import pickle

columnas_entrenamiento = X.columns.tolist()  # Suponiendo que X es tu DataFrame
with open("columnas_entrenamiento.pkl", "wb") as f:
    pickle.dump(columnas_entrenamiento, f)
