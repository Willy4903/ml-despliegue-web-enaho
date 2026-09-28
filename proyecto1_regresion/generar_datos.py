"""
Proyecto 1 (Regresión): generación de una base SIMULADA con la estructura de la
ENAHO (INEI), módulo Empleo e Ingresos.

Unidad de análisis: persona ocupada (18 a 65 años).
Variable objetivo: ingreso_laboral_mensual (soles corrientes).

La base es sintética. Las relaciones (educación, experiencia, área, región, sector,
formalidad) siguen patrones conocidos del mercado laboral peruano, pero los valores
no corresponden a personas reales. Para usar microdatos reales, descargar la ENAHO
del portal de microdatos del INEI y reemplazar data/enaho_ingresos_sim.csv respetando
los nombres de columnas.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEMILLA = 2026
N = 8000
RUTA = Path(__file__).parent / "data" / "enaho_ingresos_sim.csv"

NIVELES = [
    "Primaria o menos",
    "Secundaria",
    "Superior no universitaria",
    "Superior universitaria",
    "Posgrado",
]
RANGO_ANIOS = {
    "Primaria o menos": (2, 6),
    "Secundaria": (7, 11),
    "Superior no universitaria": (12, 15),
    "Superior universitaria": (14, 17),
    "Posgrado": (17, 20),
}
SECTORES = [
    "Agropecuario",
    "Manufactura",
    "Construccion",
    "Comercio",
    "Servicios",
    "Administracion publica, educacion y salud",
]
CATEGORIAS = ["Empleado", "Obrero", "Trabajador independiente", "Empleador"]
TAMANIOS = ["1 a 10 trabajadores", "11 a 50 trabajadores", "Mas de 50 trabajadores"]
REGIONES = ["Lima Metropolitana", "Costa", "Sierra", "Selva"]


def generar(n: int = N, semilla: int = SEMILLA) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)

    area = rng.choice(["Urbana", "Rural"], size=n, p=[0.75, 0.25])
    urbano = area == "Urbana"

    region = np.empty(n, dtype=object)
    region[urbano] = rng.choice(REGIONES, size=urbano.sum(), p=[0.38, 0.27, 0.24, 0.11])
    region[~urbano] = rng.choice(
        ["Costa", "Sierra", "Selva"], size=(~urbano).sum(), p=[0.15, 0.60, 0.25]
    )

    sexo = rng.choice(["Hombre", "Mujer"], size=n, p=[0.56, 0.44])
    edad = np.clip(rng.normal(39, 12, n), 18, 65).round().astype(int)

    nivel = np.empty(n, dtype=object)
    nivel[urbano] = rng.choice(NIVELES, size=urbano.sum(), p=[0.10, 0.38, 0.22, 0.27, 0.03])
    nivel[~urbano] = rng.choice(NIVELES, size=(~urbano).sum(), p=[0.42, 0.42, 0.08, 0.07, 0.01])

    anios_estudio = np.array(
        [rng.integers(RANGO_ANIOS[x][0], RANGO_ANIOS[x][1] + 1) for x in nivel]
    )
    experiencia = np.clip(edad - anios_estudio - 6 + rng.normal(0, 2, n), 0, 45).round().astype(int)

    # Sector segun area
    sector = np.empty(n, dtype=object)
    sector[urbano] = rng.choice(
        SECTORES, size=urbano.sum(), p=[0.03, 0.14, 0.09, 0.26, 0.30, 0.18]
    )
    sector[~urbano] = rng.choice(
        SECTORES, size=(~urbano).sum(), p=[0.70, 0.05, 0.05, 0.08, 0.07, 0.05]
    )

    # Categoria ocupacional
    categoria = np.empty(n, dtype=object)
    for i in range(n):
        if urbano[i]:
            p = [0.42, 0.16, 0.38, 0.04]
        else:
            p = [0.10, 0.14, 0.72, 0.04]
        if nivel[i] in ("Superior universitaria", "Posgrado"):
            p = [0.62, 0.05, 0.28, 0.05]
        categoria[i] = rng.choice(CATEGORIAS, p=p)

    # Tamanio de empresa
    tamanio = np.empty(n, dtype=object)
    for i in range(n):
        if categoria[i] == "Trabajador independiente":
            p = [0.93, 0.05, 0.02]
        elif categoria[i] == "Empleador":
            p = [0.70, 0.25, 0.05]
        else:
            p = [0.35, 0.28, 0.37] if urbano[i] else [0.60, 0.25, 0.15]
        tamanio[i] = rng.choice(TAMANIOS, p=p)

    horas = np.clip(rng.normal(44, 12, n) - np.where(urbano, 0, 5), 8, 84).round().astype(int)
    n_hogar = np.clip(rng.poisson(3.3, n) + 1, 1, 10)

    # Aporte a pension (proxy de formalidad)
    logit_pens = (
        -2.0
        + 1.9 * (tamanio == "Mas de 50 trabajadores")
        + 0.8 * (tamanio == "11 a 50 trabajadores")
        + 1.0 * np.isin(nivel, ["Superior universitaria", "Posgrado"])
        + 0.4 * (nivel == "Superior no universitaria")
        + 0.7 * (categoria == "Empleado")
        - 1.0 * (categoria == "Trabajador independiente")
        - 0.8 * (~urbano)
        + 0.9 * (sector == "Administracion publica, educacion y salud")
    )
    aporta = np.where(rng.random(n) < 1 / (1 + np.exp(-logit_pens)), "Si", "No")

    # Ingreso laboral (log-normal con estructura)
    ef_region = {"Lima Metropolitana": 0.18, "Costa": 0.02, "Sierra": -0.10, "Selva": 0.0}
    ef_sector = {
        "Agropecuario": -0.25,
        "Manufactura": 0.0,
        "Construccion": 0.05,
        "Comercio": -0.05,
        "Servicios": 0.03,
        "Administracion publica, educacion y salud": 0.22,
    }
    ef_cat = {"Empleado": 0.12, "Obrero": -0.08, "Trabajador independiente": -0.10, "Empleador": 0.35}
    ef_tam = {"1 a 10 trabajadores": -0.10, "11 a 50 trabajadores": 0.05, "Mas de 50 trabajadores": 0.22}
    ef_niv = {
        "Primaria o menos": 0.0,
        "Secundaria": 0.0,
        "Superior no universitaria": 0.0,
        "Superior universitaria": 0.10,
        "Posgrado": 0.30,
    }

    log_ing = (
        5.72
        + 0.062 * anios_estudio
        + 0.022 * experiencia
        - 0.00035 * experiencia**2
        + 0.35 * np.log(horas / 40)
        - 0.28 * (~urbano)
        - 0.10 * (sexo == "Mujer")
        + 0.18 * (aporta == "Si")
        + np.array([ef_region[x] for x in region])
        + np.array([ef_sector[x] for x in sector])
        + np.array([ef_cat[x] for x in categoria])
        + np.array([ef_tam[x] for x in tamanio])
        + np.array([ef_niv[x] for x in nivel])
        # Interacciones y efectos no lineales (rendimientos distintos segun contexto)
        + 0.045 * anios_estudio * urbano
        + 0.16 * ((nivel == "Superior universitaria") & (region == "Lima Metropolitana"))
        + 0.22 * ((categoria == "Empleador") & (tamanio != "1 a 10 trabajadores"))
        + 0.12 * ((aporta == "Si") & (tamanio == "Mas de 50 trabajadores"))
        + 0.12 * ((sector == "Administracion publica, educacion y salud") & (region == "Sierra"))
        - 0.16 * ((sector == "Agropecuario") & (categoria == "Trabajador independiente"))
        - 0.12 * (horas > 60)
        + 0.10 * ((experiencia > 10) & (experiencia < 30) & (anios_estudio >= 14))
        + rng.normal(0, 0.30, n)
    )
    ingreso = np.clip(np.exp(log_ing), 150, 15000).round(0)

    df = pd.DataFrame(
        {
            "edad": edad,
            "sexo": sexo,
            "area": area,
            "region_natural": region,
            "nivel_educativo": nivel,
            "anios_estudio": anios_estudio,
            "experiencia_anios": experiencia,
            "horas_semana": horas,
            "categoria_ocupacional": categoria,
            "sector": sector,
            "tamanio_empresa": tamanio,
            "aporta_pension": aporta,
            "miembros_hogar": n_hogar,
            "ingreso_laboral_mensual": ingreso,
        }
    )

    # Valores perdidos realistas (~1.5%) para justificar la imputacion en el pipeline
    for col, tasa in [("horas_semana", 0.015), ("aporta_pension", 0.01), ("experiencia_anios", 0.01)]:
        df.loc[rng.random(n) < tasa, col] = np.nan
    return df


if __name__ == "__main__":
    RUTA.parent.mkdir(parents=True, exist_ok=True)
    datos = generar()
    datos.to_csv(RUTA, index=False, encoding="utf-8")
    print(f"Base guardada en {RUTA} con forma {datos.shape}")
    print(datos["ingreso_laboral_mensual"].describe().round(0))
    print(datos.isna().mean().round(3)[datos.isna().any()])
