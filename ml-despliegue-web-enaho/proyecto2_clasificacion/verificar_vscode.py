"""
Proyecto 2 (Clasificación) - Verificación desde VS Code.

Ejecutar en la terminal de VS Code:   python verificar_vscode.py

Imprime la probabilidad de pobreza y la clase predicha para 5 hogares fijos. Los valores
deben ser IDÉNTICOS a los de la pestaña "Verificación" de la app desplegada en Streamlit
y a los guardados durante el entrenamiento (modelo/metricas.json).
"""
import pandas as pd

from modelo_utils import CASOS_VERIFICACION, cargar_metricas, cargar_modelo, clasificar, probabilidad


def main():
    modelo = cargar_modelo()
    met = cargar_metricas()
    prob = probabilidad(modelo, CASOS_VERIFICACION)
    tabla = pd.DataFrame(
        {
            "Caso": [c["nombre"] for c in CASOS_VERIFICACION],
            "Probabilidad VS Code": prob,
            "Esperada entrenamiento": met["casos_verificacion_esperados"],
            "Clase (umbral óptimo)": clasificar(prob, met["umbral_optimo"]),
        }
    )
    tabla["Coincide"] = (tabla["Probabilidad VS Code"] - tabla["Esperada entrenamiento"]).abs() < 1e-4

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print(f"Modelo: {met['modelo_seleccionado']} | umbral óptimo: {met['umbral_optimo']:.2f} | "
          f"scikit-learn {met['sklearn_version']} (entrenamiento)")
    print(tabla.to_string(index=False))
    print()
    if tabla["Coincide"].all():
        print("VERIFICACIÓN CORRECTA: las 5 probabilidades coinciden con las del entrenamiento.")
        print("Compare estos valores con la pestaña 'Verificación' de la app en Streamlit.")
    else:
        raise SystemExit("Hay diferencias: revise la versión de scikit-learn (requirements.txt).")


if __name__ == "__main__":
    main()
