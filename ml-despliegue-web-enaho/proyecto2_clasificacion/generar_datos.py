"""
Proyecto 2 (Clasificacion): generacion de una base SIMULADA con la estructura de la
ENAHO (INEI), modulos de Caracteristicas de la vivienda y del hogar, Educacion y Empleo.

Unidad de analisis: hogar (caracteristicas del jefe del hogar y de la vivienda).
Variable objetivo dicotomica: pobre_monetario (1 = hogar en pobreza monetaria, 0 = no pobre).

La base es sintetica: replica asociaciones conocidas (area rural, educacion del jefe,
tamanio del hogar, acceso a servicios, formalidad del empleo) pero no contiene hogares
reales. Para usar microdatos reales, descargar la ENAHO del portal de microdatos del INEI
y reemplazar data/enaho_pobreza_sim.csv conservando los nombres de columnas.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEMILLA = 2026
N = 10000
RUTA = Path(__file__).parent / "data" / "enaho_pobreza_sim.csv"

NIVELES = [
    "Primaria o menos",
    "Secundaria",
    "Superior no universitaria",
    "Superior universitaria",
    "Posgrado",
]
REGIONES = ["Lima Metropolitana", "Costa", "Sierra", "Selva"]
OCUPACIONES = ["Empleo formal", "Empleo informal", "Sin empleo"]
PISOS = ["Tierra", "Cemento", "Loseta, parquet o similar"]
TENENCIAS = ["Propia", "Alquilada", "Cedida u otra"]


def generar(n: int = N, semilla: int = SEMILLA) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)

    area = rng.choice(["Urbana", "Rural"], size=n, p=[0.72, 0.28])
    urbano = area == "Urbana"

    region = np.empty(n, dtype=object)
    region[urbano] = rng.choice(REGIONES, size=urbano.sum(), p=[0.38, 0.27, 0.24, 0.11])
    region[~urbano] = rng.choice(["Costa", "Sierra", "Selva"], size=(~urbano).sum(), p=[0.15, 0.60, 0.25])

    edad = np.clip(rng.normal(48, 14, n), 22, 85).round().astype(int)
    sexo = rng.choice(["Hombre", "Mujer"], size=n, p=[0.70, 0.30])

    nivel = np.empty(n, dtype=object)
    nivel[urbano] = rng.choice(NIVELES, size=urbano.sum(), p=[0.16, 0.38, 0.20, 0.23, 0.03])
    nivel[~urbano] = rng.choice(NIVELES, size=(~urbano).sum(), p=[0.52, 0.36, 0.06, 0.05, 0.01])

    tamanio = np.clip(rng.poisson(2.9, n) + 1, 1, 10)
    menores = np.minimum(rng.binomial(np.maximum(tamanio - 1, 0), 0.38), 6)
    perceptores = np.clip(rng.binomial(np.maximum(tamanio - menores, 1), 0.45), 0, 5)
    perceptores = np.maximum(perceptores, 1 * (rng.random(n) < 0.9))
    habitaciones = np.clip(rng.poisson(2.6, n) + 1 + (urbano * 0), 1, 8)

    ocup = np.empty(n, dtype=object)
    for i in range(n):
        if nivel[i] in ("Superior universitaria", "Posgrado"):
            p = [0.62, 0.30, 0.08]
        elif nivel[i] == "Superior no universitaria":
            p = [0.42, 0.46, 0.12]
        elif urbano[i]:
            p = [0.20, 0.62, 0.18]
        else:
            p = [0.05, 0.80, 0.15]
        if edad[i] > 65:
            p = [p[0] * 0.5, p[1] * 0.8, p[2] + 0.35]
            s = sum(p)
            p = [x / s for x in p]
        ocup[i] = rng.choice(OCUPACIONES, p=p)

    agua = np.where(rng.random(n) < np.where(urbano, 0.94, 0.62), "Si", "No")
    elec = np.where(rng.random(n) < np.where(urbano, 0.99, 0.86), "Si", "No")
    internet = np.where(rng.random(n) < np.where(urbano, 0.62, 0.14) - 0.15 * (nivel == "Primaria o menos"), "Si", "No")
    piso = np.empty(n, dtype=object)
    piso[urbano] = rng.choice(PISOS, size=urbano.sum(), p=[0.05, 0.60, 0.35])
    piso[~urbano] = rng.choice(PISOS, size=(~urbano).sum(), p=[0.48, 0.44, 0.08])
    tenencia = rng.choice(TENENCIAS, size=n, p=[0.68, 0.16, 0.16])

    ef_reg = {"Lima Metropolitana": -0.45, "Costa": 0.0, "Sierra": 0.25, "Selva": 0.35}
    ef_niv = {
        "Primaria o menos": 0.75, "Secundaria": 0.20, "Superior no universitaria": -0.30,
        "Superior universitaria": -0.90, "Posgrado": -1.50,
    }
    ef_ocup = {"Empleo formal": -0.70, "Empleo informal": 0.15, "Sin empleo": 1.00}

    z = (
        -2.75
        + 0.70 * (~urbano)
        + np.array([ef_reg[x] for x in region])
        + np.array([ef_niv[x] for x in nivel])
        + np.array([ef_ocup[x] for x in ocup])
        + 0.30 * (tamanio - 4)
        + 0.32 * menores
        - 0.55 * (perceptores - 1)
        + 0.015 * np.abs(edad - 45)
        - 0.15 * (habitaciones - 3)
        + 0.50 * (agua == "No")
        + 0.45 * (elec == "No")
        - 0.70 * (internet == "Si")
        + 0.60 * (piso == "Tierra")
        - 0.30 * (piso == "Loseta, parquet o similar")
        + 0.15 * (tenencia == "Alquilada")
        # Interacciones y umbrales (relaciones no lineales)
        + 1.20 * ((~urbano) & (agua == "No") & (elec == "No"))
        + 1.10 * ((tamanio >= 6) & (perceptores <= 1))
        + 1.30 * ((menores / np.maximum(tamanio, 1)) > 0.5)
        + 1.10 * ((ocup == "Sin empleo") & (edad > 62) & np.isin(nivel, ["Primaria o menos", "Secundaria"]))
        + 0.90 * ((region == "Selva") & (~urbano) & (piso == "Tierra"))
        - 0.90 * ((internet == "Si") & np.isin(nivel, ["Superior no universitaria", "Superior universitaria", "Posgrado"]))
        - 1.00 * ((ocup == "Empleo formal") & (perceptores >= 2))
        + 0.90 * ((tenencia == "Alquilada") & (perceptores <= 1) & (tamanio >= 4))
        + rng.normal(0, 0.35, n)
    )
    prob = 1 / (1 + np.exp(-z))
    pobre = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame(
        {
            "area": area,
            "region_natural": region,
            "edad_jefe": edad,
            "sexo_jefe": sexo,
            "nivel_educativo_jefe": nivel,
            "ocupacion_jefe": ocup,
            "tamanio_hogar": tamanio,
            "menores_14": menores,
            "perceptores_ingreso": perceptores,
            "habitaciones": habitaciones,
            "agua_red_publica": agua,
            "electricidad": elec,
            "internet": internet,
            "material_piso": piso,
            "tenencia_vivienda": tenencia,
            "pobre_monetario": pobre,
        }
    )
    for col, tasa in [("habitaciones", 0.01), ("material_piso", 0.008), ("internet", 0.01)]:
        df.loc[rng.random(n) < tasa, col] = np.nan
    return df


if __name__ == "__main__":
    RUTA.parent.mkdir(parents=True, exist_ok=True)
    datos = generar()
    datos.to_csv(RUTA, index=False, encoding="utf-8")
    print(f"Base guardada en {RUTA} con forma {datos.shape}")
    print("Prevalencia de pobreza:", round(datos["pobre_monetario"].mean(), 3))
    print(datos.groupby("area")["pobre_monetario"].mean().round(3))
