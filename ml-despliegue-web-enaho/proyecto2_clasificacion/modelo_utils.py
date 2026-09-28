"""
Funciones compartidas por entrenar.py, app.py y verificar_vscode.py (Proyecto 2).

La app de Streamlit y el script de VS Code usan estas mismas funciones y el mismo
archivo .joblib, por lo que las probabilidades coinciden por construccion.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).parent
RUTA_DATOS = BASE / "data" / "enaho_pobreza_sim.csv"
RUTA_MODELO = BASE / "modelo" / "modelo_clasificacion.joblib"
RUTA_METRICAS = BASE / "modelo" / "metricas.json"

OBJETIVO = "pobre_monetario"  # dicotomica: 1 = pobre, 0 = no pobre

# Distincion entre variables numericas y categoricas
COLS_NUM = ["edad_jefe", "tamanio_hogar", "menores_14", "perceptores_ingreso", "habitaciones"]
COLS_CAT = [
    "area",
    "region_natural",
    "sexo_jefe",
    "nivel_educativo_jefe",
    "ocupacion_jefe",
    "agua_red_publica",
    "electricidad",
    "internet",
    "material_piso",
    "tenencia_vivienda",
]
FEATURES = COLS_NUM + COLS_CAT

CASOS_VERIFICACION = [
    {
        "nombre": "Caso 1: hogar urbano con jefe universitario",
        "area": "Urbana", "region_natural": "Lima Metropolitana", "edad_jefe": 42, "sexo_jefe": "Hombre",
        "nivel_educativo_jefe": "Superior universitaria", "ocupacion_jefe": "Empleo formal",
        "tamanio_hogar": 4, "menores_14": 2, "perceptores_ingreso": 2, "habitaciones": 4,
        "agua_red_publica": "Si", "electricidad": "Si", "internet": "Si",
        "material_piso": "Loseta, parquet o similar", "tenencia_vivienda": "Propia",
    },
    {
        "nombre": "Caso 2: hogar rural numeroso en la sierra",
        "area": "Rural", "region_natural": "Sierra", "edad_jefe": 55, "sexo_jefe": "Hombre",
        "nivel_educativo_jefe": "Primaria o menos", "ocupacion_jefe": "Empleo informal",
        "tamanio_hogar": 7, "menores_14": 4, "perceptores_ingreso": 1, "habitaciones": 2,
        "agua_red_publica": "No", "electricidad": "Si", "internet": "No",
        "material_piso": "Tierra", "tenencia_vivienda": "Propia",
    },
    {
        "nombre": "Caso 3: hogar urbano de ingreso medio",
        "area": "Urbana", "region_natural": "Costa", "edad_jefe": 38, "sexo_jefe": "Mujer",
        "nivel_educativo_jefe": "Secundaria", "ocupacion_jefe": "Empleo informal",
        "tamanio_hogar": 5, "menores_14": 2, "perceptores_ingreso": 1, "habitaciones": 3,
        "agua_red_publica": "Si", "electricidad": "Si", "internet": "Si",
        "material_piso": "Cemento", "tenencia_vivienda": "Alquilada",
    },
    {
        "nombre": "Caso 4: hogar de la selva sin servicios",
        "area": "Rural", "region_natural": "Selva", "edad_jefe": 61, "sexo_jefe": "Mujer",
        "nivel_educativo_jefe": "Primaria o menos", "ocupacion_jefe": "Sin empleo",
        "tamanio_hogar": 6, "menores_14": 3, "perceptores_ingreso": 1, "habitaciones": 2,
        "agua_red_publica": "No", "electricidad": "No", "internet": "No",
        "material_piso": "Tierra", "tenencia_vivienda": "Cedida u otra",
    },
    {
        "nombre": "Caso 5: adulto mayor urbano con posgrado",
        "area": "Urbana", "region_natural": "Sierra", "edad_jefe": 70, "sexo_jefe": "Hombre",
        "nivel_educativo_jefe": "Posgrado", "ocupacion_jefe": "Sin empleo",
        "tamanio_hogar": 2, "menores_14": 0, "perceptores_ingreso": 1, "habitaciones": 5,
        "agua_red_publica": "Si", "electricidad": "Si", "internet": "Si",
        "material_piso": "Loseta, parquet o similar", "tenencia_vivienda": "Propia",
    },
]


def construir_preprocesador(escalar: bool = False) -> ColumnTransformer:
    pasos_num = [("imputar", SimpleImputer(strategy="median"))]
    if escalar:
        pasos_num.append(("escalar", StandardScaler()))
    num = Pipeline(pasos_num)
    cat = Pipeline(
        [
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer([("num", num, COLS_NUM), ("cat", cat, COLS_CAT)])


def cargar_datos() -> pd.DataFrame:
    return pd.read_csv(RUTA_DATOS)


def cargar_modelo():
    """Carga el .joblib; si la version de scikit-learn no coincide, reentrena con la misma semilla."""
    try:
        return joblib.load(RUTA_MODELO)
    except Exception:
        from entrenar import entrenar_y_guardar

        entrenar_y_guardar(verbose=False)
        return joblib.load(RUTA_MODELO)


def cargar_metricas() -> dict:
    with open(RUTA_METRICAS, encoding="utf-8") as f:
        return json.load(f)


def probabilidad(modelo, filas) -> np.ndarray:
    """Probabilidad de pobreza monetaria (clase 1). `filas` es un dict o una lista de dicts."""
    if isinstance(filas, dict):
        filas = [filas]
    df = pd.DataFrame(filas)[FEATURES]
    return np.round(modelo.predict_proba(df)[:, 1], 4)


def clasificar(prob, umbral: float) -> np.ndarray:
    return (np.asarray(prob) >= umbral).astype(int)


def explicacion_local(modelo, fila: dict, referencia: dict) -> pd.DataFrame:
    """
    Cambio en la probabilidad de pobreza al reemplazar cada variable por su valor de
    referencia (mediana o moda de la base). Positivo: la variable eleva el riesgo.
    """
    base = probabilidad(modelo, fila)[0]
    alternos = []
    for col in FEATURES:
        alt = dict(fila)
        alt[col] = referencia[col]
        alternos.append(alt)
    sin = probabilidad(modelo, alternos)
    out = pd.DataFrame({"variable": FEATURES, "efecto_prob": base - sin})
    out["valor"] = [fila[c] for c in FEATURES]
    return out.sort_values("efecto_prob", key=np.abs, ascending=False).reset_index(drop=True)
