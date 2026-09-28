"""
Proyecto 1 (Regresion) - Verificación desde VS Code.

Ejecutar en la terminal de VS Code:   python verificar_vscode.py

Imprime la prediccion de 5 casos fijos. Los montos deben ser IDENTICOS a los de la
pestaña "Verificación" de la app desplegada en Streamlit, y a los valores guardados
durante el entrenamiento (modelo/metricas.json).

Tambien acepta un caso propio, por ejemplo:
    python verificar_vscode.py --edad 30 --sexo Hombre --area Urbana
"""
import argparse

import pandas as pd

from modelo_utils import (
    CASOS_VERIFICACION, FEATURES, cargar_metricas, cargar_modelo, predecir,
)


def main():
    modelo = cargar_modelo()
    met = cargar_metricas()

    pred = predecir(modelo, CASOS_VERIFICACION)
    tabla = pd.DataFrame(
        {
            "Caso": [c["nombre"] for c in CASOS_VERIFICACION],
            "Predicción VS Code (S/)": pred,
            "Esperado entrenamiento (S/)": met["casos_verificacion_esperados"],
        }
    )
    tabla["Coincide"] = (tabla.iloc[:, 1] - tabla.iloc[:, 2]).abs() < 0.01

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print(f"Modelo: {met['modelo_seleccionado']}  |  scikit-learn {met['sklearn_version']} (entrenamiento)")
    print(tabla.to_string(index=False))
    print()
    if tabla["Coincide"].all():
        print("VERIFICACIÓN CORRECTA: las 5 predicciones coinciden con las del entrenamiento.")
        print("Compare estos montos con la pestaña 'Verificación' de la app en Streamlit.")
    else:
        raise SystemExit("Hay diferencias: revise la version de scikit-learn (requirements.txt).")

    # Caso libre opcional: se parte del Caso 1 y se sobrescriben los campos indicados
    ap = argparse.ArgumentParser()
    for f in FEATURES:
        ap.add_argument(f"--{f}")
    args, _ = ap.parse_known_args()
    cambios = {k: v for k, v in vars(args).items() if v is not None}
    if cambios:
        caso = dict(CASOS_VERIFICACION[0])
        for k, v in cambios.items():
            caso[k] = float(v) if k in ("edad", "anios_estudio", "experiencia_anios", "horas_semana", "miembros_hogar") else v
        print(f"\nCaso personalizado {cambios}: S/ {predecir(modelo, caso)[0]:,.2f}")


if __name__ == "__main__":
    main()
