"""
Proyecto 1 (Regresion): entrenamiento, comparacion y seleccion del mejor modelo.

Ejecutar:  python entrenar.py
Genera en modelo/: modelo_regresion.joblib, metricas.json y los graficos PNG.
"""
import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from modelo_utils import (
    BASE, CASOS_VERIFICACION, COLS_CAT, COLS_NUM, FEATURES, OBJETIVO, RUTA_METRICAS,
    RUTA_MODELO, cargar_datos, construir_preprocesador, envolver_log, predecir,
)

SEMILLA = 42
NAVY, GOLD = "#0D3B66", "#E8A537"


def _pipeline(regresor, escalar=False):
    return envolver_log(
        Pipeline([("prep", construir_preprocesador(escalar)), ("modelo", regresor)])
    )


def candidatos():
    """Modelo base + los dos modelos exigidos (Random Forest y Gradient Boosting)."""
    return {
        "Ridge (base)": (
            _pipeline(Ridge(), escalar=True),
            {"regressor__modelo__alpha": [0.1, 1.0, 5.0, 10.0, 50.0]},
        ),
        "Random Forest": (
            _pipeline(RandomForestRegressor(random_state=SEMILLA, n_jobs=-1)),
            {
                "regressor__modelo__n_estimators": [120, 200],
                "regressor__modelo__max_depth": [8, 12, 16],
                "regressor__modelo__min_samples_leaf": [1, 3, 5, 10],
                "regressor__modelo__max_features": [0.5, 0.8, 1.0],
            },
        ),
        "Gradient Boosting": (
            _pipeline(GradientBoostingRegressor(random_state=SEMILLA)),
            {
                "regressor__modelo__n_estimators": [150, 250, 350],
                "regressor__modelo__learning_rate": [0.03, 0.05, 0.1],
                "regressor__modelo__max_depth": [2, 3, 4],
                "regressor__modelo__subsample": [0.7, 0.85, 1.0],
                "regressor__modelo__min_samples_leaf": [5, 10, 20],
            },
        ),
    }


def evaluar(modelo, X, y):
    pred = modelo.predict(X)
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)),
    }


def entrenar_y_guardar(verbose: bool = True) -> dict:
    df = cargar_datos()
    X, y = df[FEATURES], df[OBJETIVO]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=SEMILLA)

    resultados, ajustados = {}, {}
    for nombre, (pipe, grilla) in candidatos().items():
        t0 = time.time()
        n_iter = min(6, int(np.prod([len(v) for v in grilla.values()])))
        busq = RandomizedSearchCV(
            pipe, grilla, n_iter=n_iter, cv=3, scoring="neg_root_mean_squared_error",
            random_state=SEMILLA, n_jobs=-1, refit=True,
        )
        busq.fit(X_tr, y_tr)
        m_te = evaluar(busq.best_estimator_, X_te, y_te)
        m_tr = evaluar(busq.best_estimator_, X_tr, y_tr)
        resultados[nombre] = {
            "cv_RMSE": float(-busq.best_score_),
            "test_MAE": m_te["MAE"], "test_RMSE": m_te["RMSE"], "test_R2": m_te["R2"],
            "train_R2": m_tr["R2"],
            "mejores_parametros": {k.split("__")[-1]: (None if v is None else (v.item() if hasattr(v, "item") else v))
                                    for k, v in busq.best_params_.items()},
            "segundos": round(time.time() - t0, 1),
        }
        ajustados[nombre] = busq.best_estimator_
        if verbose:
            r = resultados[nombre]
            print(f"{nombre:20s} cvRMSE={r['cv_RMSE']:8.1f}  testRMSE={r['test_RMSE']:8.1f}  "
                  f"MAE={r['test_MAE']:8.1f}  R2={r['test_R2']:.3f}  ({r['segundos']}s)")

    # Seleccion por RMSE de validacion cruzada (el test no interviene en la decision)
    mejor = min(resultados, key=lambda k: resultados[k]["cv_RMSE"])
    modelo = ajustados[mejor]
    if verbose:
        print("Modelo seleccionado:", mejor)

    # Importancia de variables (permutacion sobre el conjunto de prueba, variables originales)
    imp = permutation_importance(
        modelo, X_te, y_te, n_repeats=10, random_state=SEMILLA,
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    importancia = (
        pd.DataFrame({"variable": FEATURES, "importancia": imp.importances_mean, "sd": imp.importances_std})
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )

    # Referencia (mediana / moda) para la explicacion local de la app
    referencia = {c: float(df[c].median()) for c in COLS_NUM}
    referencia.update({c: df[c].mode().iloc[0] for c in COLS_CAT})
    referencia["edad"] = int(round(referencia["edad"]))
    for c in ("anios_estudio", "experiencia_anios", "horas_semana", "miembros_hogar"):
        referencia[c] = int(round(referencia[c]))

    esperado = predecir(modelo, CASOS_VERIFICACION).tolist()

    joblib.dump(modelo, RUTA_MODELO, compress=3)
    metricas = {
        "modelo_seleccionado": mejor,
        "criterio": "menor RMSE en validacion cruzada (3 folds) sobre el 80% de entrenamiento",
        "resultados": resultados,
        "importancia": importancia.to_dict(orient="records"),
        "referencia": referencia,
        "n_entrenamiento": int(len(X_tr)),
        "n_prueba": int(len(X_te)),
        "semilla": SEMILLA,
        "sklearn_version": sklearn.__version__,
        "casos_verificacion_esperados": esperado,
    }
    with open(RUTA_METRICAS, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)

    _graficos(modelo, mejor, importancia, X_te, y_te, resultados, df)
    if verbose:
        print(importancia.head(8).round(1).to_string(index=False))
        print("Predicciones esperadas (casos de verificacion):", esperado)
    return metricas


def _graficos(modelo, mejor, importancia, X_te, y_te, resultados, df):
    out = BASE / "modelo"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    top = importancia.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(top["variable"], top["importancia"], xerr=top["sd"], color=NAVY, ecolor=GOLD)
    ax.set_xlabel("Aumento del RMSE al permutar la variable (soles)")
    ax.set_title(f"Importancia de variables ({mejor})")
    fig.tight_layout()
    fig.savefig(out / "importancia_variables.png", dpi=150)
    plt.close(fig)

    pred = modelo.predict(X_te)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.scatter(y_te, pred, s=6, alpha=0.35, color=NAVY)
    lim = [0, float(max(y_te.max(), pred.max()))]
    ax.plot(lim, lim, color=GOLD, lw=2)
    ax.set_xlabel("Ingreso real (soles)")
    ax.set_ylabel("Ingreso predicho (soles)")
    ax.set_title("Real vs predicho (conjunto de prueba)")
    fig.tight_layout()
    fig.savefig(out / "real_vs_predicho.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(df[OBJETIVO], bins=50, color=NAVY)
    ax.axvline(df[OBJETIVO].median(), color=GOLD, lw=2, label=f"Mediana: S/ {df[OBJETIVO].median():,.0f}")
    ax.set_xlabel("Ingreso laboral mensual (soles)")
    ax.set_ylabel("Personas")
    ax.set_title("Distribución del ingreso laboral")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "distribucion_ingreso.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    entrenar_y_guardar()
