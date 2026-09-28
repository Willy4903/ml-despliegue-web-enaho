"""
Funciones compartidas por entrenar.py, app.py y verificar_vscode.py.

Como la app de Streamlit y el script de VS Code usan exactamente estas mismas
funciones y el mismo archivo .joblib, las predicciones coinciden por construccion.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).parent
RUTA_DATOS = BASE / "data" / "enaho_ingresos_sim.csv"
RUTA_MODELO = BASE / "modelo" / "modelo_regresion.joblib"
RUTA_METRICAS = BASE / "modelo" / "metricas.json"

OBJETIVO = "ingreso_laboral_mensual"

# Distincion entre variables numericas y categoricas (punto 1 del proyecto)
COLS_NUM = ["edad", "anios_estudio", "experiencia_anios", "horas_semana", "miembros_hogar"]
COLS_CAT = [
    "sexo",
    "area",
    "region_natural",
    "nivel_educativo",
    "categoria_ocupacional",
    "sector",
    "tamanio_empresa",
    "aporta_pension",
]
FEATURES = COLS_NUM + COLS_CAT

# Casos fijos que se comparan entre VS Code y la app desplegada
CASOS_VERIFICACION = [
    {
        "nombre": "Caso 1: profesional en Lima",
        "edad": 35, "sexo": "Mujer", "area": "Urbana", "region_natural": "Lima Metropolitana",
        "nivel_educativo": "Superior universitaria", "anios_estudio": 16, "experiencia_anios": 12,
        "horas_semana": 45, "categoria_ocupacional": "Empleado",
        "sector": "Servicios", "tamanio_empresa": "Mas de 50 trabajadores",
        "aporta_pension": "Si", "miembros_hogar": 4,
    },
    {
        "nombre": "Caso 2: agricultor rural en la sierra",
        "edad": 48, "sexo": "Hombre", "area": "Rural", "region_natural": "Sierra",
        "nivel_educativo": "Primaria o menos", "anios_estudio": 5, "experiencia_anios": 37,
        "horas_semana": 50, "categoria_ocupacional": "Trabajador independiente",
        "sector": "Agropecuario", "tamanio_empresa": "1 a 10 trabajadores",
        "aporta_pension": "No", "miembros_hogar": 6,
    },
    {
        "nombre": "Caso 3: tecnico en comercio, costa urbana",
        "edad": 29, "sexo": "Hombre", "area": "Urbana", "region_natural": "Costa",
        "nivel_educativo": "Superior no universitaria", "anios_estudio": 14, "experiencia_anios": 9,
        "horas_semana": 48, "categoria_ocupacional": "Obrero",
        "sector": "Comercio", "tamanio_empresa": "11 a 50 trabajadores",
        "aporta_pension": "Si", "miembros_hogar": 3,
    },
    {
        "nombre": "Caso 4: posgrado en administracion publica",
        "edad": 44, "sexo": "Mujer", "area": "Urbana", "region_natural": "Lima Metropolitana",
        "nivel_educativo": "Posgrado", "anios_estudio": 19, "experiencia_anios": 19,
        "horas_semana": 40, "categoria_ocupacional": "Empleado",
        "sector": "Administracion publica, educacion y salud",
        "tamanio_empresa": "Mas de 50 trabajadores", "aporta_pension": "Si", "miembros_hogar": 3,
    },
    {
        "nombre": "Caso 5: empleador en la selva",
        "edad": 52, "sexo": "Hombre", "area": "Urbana", "region_natural": "Selva",
        "nivel_educativo": "Secundaria", "anios_estudio": 11, "experiencia_anios": 35,
        "horas_semana": 60, "categoria_ocupacional": "Empleador",
        "sector": "Construccion", "tamanio_empresa": "11 a 50 trabajadores",
        "aporta_pension": "No", "miembros_hogar": 5,
    },
]


def construir_preprocesador(escalar: bool = False) -> ColumnTransformer:
    """Imputacion + codificacion. El escalado solo se usa en el modelo lineal base."""
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


def envolver_log(regresor) -> TransformedTargetRegressor:
    """El ingreso es asimetrico: se modela log(1 + y) y se devuelve a soles."""
    return TransformedTargetRegressor(regressor=regresor, func=np.log1p, inverse_func=np.expm1)


def cargar_datos() -> pd.DataFrame:
    return pd.read_csv(RUTA_DATOS)


def cargar_modelo():
    """Carga el .joblib. Si la version de scikit-learn no coincide, reentrena con la misma semilla."""
    try:
        return joblib.load(RUTA_MODELO)
    except Exception:
        from entrenar import entrenar_y_guardar

        entrenar_y_guardar(verbose=False)
        return joblib.load(RUTA_MODELO)


def cargar_metricas() -> dict:
    with open(RUTA_METRICAS, encoding="utf-8") as f:
        return json.load(f)


def predecir(modelo, filas) -> np.ndarray:
    """Predice ingreso mensual (soles). `filas` es un dict o una lista de dicts."""
    if isinstance(filas, dict):
        filas = [filas]
    df = pd.DataFrame(filas)[FEATURES]
    return np.round(modelo.predict(df), 2)


def explicacion_local(modelo, fila: dict, referencia: dict) -> pd.DataFrame:
    """
    Explica una prediccion: para cada variable, cuanto cambia el ingreso predicho
    frente a reemplazar su valor por el valor de referencia (mediana o moda de la base).
    """
    base = predecir(modelo, fila)[0]
    filas = []
    for col in FEATURES:
        alt = dict(fila)
        alt[col] = referencia[col]
        filas.append(alt)
    sin = predecir(modelo, filas)
    out = pd.DataFrame({"variable": FEATURES, "efecto_soles": base - sin})
    out["valor"] = [fila[c] for c in FEATURES]
    return out.sort_values("efecto_soles", key=np.abs, ascending=False).reset_index(drop=True)
