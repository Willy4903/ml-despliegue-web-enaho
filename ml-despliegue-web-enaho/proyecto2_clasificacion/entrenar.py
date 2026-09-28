"""
Proyecto 2 (Clasificacion): entrenamiento, comparacion y seleccion del mejor modelo.

Ejecutar:  python entrenar.py
Genera en modelo/: modelo_clasificacion.joblib, metricas.json y los graficos PNG.
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from modelo_utils import (
    BASE, CASOS_VERIFICACION, COLS_CAT, COLS_NUM, FEATURES, OBJETIVO, RUTA_METRICAS,
    RUTA_MODELO, cargar_datos, construir_preprocesador, probabilidad,
)

SEMILLA = 42
NAVY, GOLD = "#0D3B66", "#E8A537"


def _pipeline(clf, escalar=False):
    return Pipeline([("prep", construir_preprocesador(escalar)), ("modelo", clf)])


def candidatos():
    """Modelo base + los dos modelos exigidos (Random Forest y Gradient Boosting)."""
    return {
        "Regresion logistica (base)": (
            _pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"), escalar=True),
            {"modelo__C": [0.05, 0.1, 0.5, 1.0, 5.0]},
        ),
        "Random Forest": (
            _pipeline(RandomForestClassifier(random_state=SEMILLA, n_jobs=-1, class_weight="balanced_subsample")),
            {
                "modelo__n_estimators": [120, 200],
                "modelo__max_depth": [6, 10, 14],
                "modelo__min_samples_leaf": [3, 5, 10, 20],
                "modelo__max_features": ["sqrt", 0.5],
            },
        ),
        "Gradient Boosting": (
            _pipeline(GradientBoostingClassifier(random_state=SEMILLA)),
            {
                "modelo__n_estimators": [100, 200, 300],
                "modelo__learning_rate": [0.03, 0.05, 0.1],
                "modelo__max_depth": [2, 3, 4],
                "modelo__subsample": [0.7, 0.85, 1.0],
                "modelo__min_samples_leaf": [10, 20, 40],
            },
        ),
    }


def _limpio(v):
    return v.item() if hasattr(v, "item") else v


def metricas_clasificacion(y, prob, umbral):
    pred = (prob >= umbral).astype(int)
    return {
        "ROC_AUC": float(roc_auc_score(y, prob)),
        "PR_AUC": float(average_precision_score(y, prob)),
        "Accuracy": float(accuracy_score(y, pred)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred)),
        "F1": float(f1_score(y, pred)),
    }


def entrenar_y_guardar(verbose: bool = True) -> dict:
    df = cargar_datos()
    X, y = df[FEATURES], df[OBJETIVO]
    assert set(y.unique()) <= {0, 1}, "El objetivo debe estar codificado con 0 y 1"
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=SEMILLA, stratify=y)

    resultados, ajustados, probas = {}, {}, {}
    for nombre, (pipe, grilla) in candidatos().items():
        t0 = time.time()
        n_iter = min(6, int(np.prod([len(v) for v in grilla.values()])))
        busq = RandomizedSearchCV(
            pipe, grilla, n_iter=n_iter, cv=3, scoring="roc_auc",
            random_state=SEMILLA, n_jobs=-1, refit=True,
        )
        busq.fit(X_tr, y_tr)
        prob_te = busq.best_estimator_.predict_proba(X_te)[:, 1]
        m = metricas_clasificacion(y_te, prob_te, 0.5)
        resultados[nombre] = {
            "cv_ROC_AUC": float(busq.best_score_),
            **{f"test_{k}": v for k, v in m.items()},
            "train_ROC_AUC": float(roc_auc_score(y_tr, busq.best_estimator_.predict_proba(X_tr)[:, 1])),
            "mejores_parametros": {k.split("__")[-1]: _limpio(v) for k, v in busq.best_params_.items()},
            "segundos": round(time.time() - t0, 1),
        }
        ajustados[nombre], probas[nombre] = busq.best_estimator_, prob_te
        if verbose:
            r = resultados[nombre]
            print(f"{nombre:28s} cvAUC={r['cv_ROC_AUC']:.4f} testAUC={r['test_ROC_AUC']:.4f} "
                  f"F1={r['test_F1']:.3f} Rec={r['test_Recall']:.3f} ({r['segundos']}s)")

    mejor = max(resultados, key=lambda k: resultados[k]["cv_ROC_AUC"])
    modelo = ajustados[mejor]

    # Umbral que maximiza F1 con predicciones fuera de muestra (solo entrenamiento)
    oof = cross_val_predict(modelo, X_tr, y_tr, cv=3, method="predict_proba", n_jobs=-1)[:, 1]
    prec, rec, thr = precision_recall_curve(y_tr, oof)
    f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    umbral = float(np.round(thr[int(np.argmax(f1s))], 2))

    prob_te = probas[mejor]
    final = metricas_clasificacion(y_te, prob_te, umbral)
    cm = confusion_matrix(y_te, (prob_te >= umbral).astype(int)).tolist()

    imp = permutation_importance(
        modelo, X_te, y_te, n_repeats=8, random_state=SEMILLA, scoring="roc_auc", n_jobs=-1
    )
    importancia = (
        pd.DataFrame({"variable": FEATURES, "importancia": imp.importances_mean, "sd": imp.importances_std})
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )

    referencia = {c: float(df[c].median()) for c in COLS_NUM}
    referencia.update({c: df[c].mode().iloc[0] for c in COLS_CAT})
    for c in COLS_NUM:
        referencia[c] = int(round(referencia[c]))

    esperado = probabilidad(modelo, CASOS_VERIFICACION).tolist()

    joblib.dump(modelo, RUTA_MODELO, compress=3)
    metricas = {
        "modelo_seleccionado": mejor,
        "criterio": "mayor ROC-AUC en validacion cruzada (3 folds) sobre el 80% de entrenamiento",
        "umbral_optimo": umbral,
        "metricas_finales_prueba": final,
        "matriz_confusion": cm,
        "resultados": resultados,
        "importancia": importancia.to_dict(orient="records"),
        "referencia": referencia,
        "prevalencia": float(y.mean()),
        "n_entrenamiento": int(len(X_tr)),
        "n_prueba": int(len(X_te)),
        "semilla": SEMILLA,
        "sklearn_version": sklearn.__version__,
        "casos_verificacion_esperados": esperado,
    }
    with open(RUTA_METRICAS, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)

    _graficos(mejor, importancia, cm, probas, y_te, df)
    if verbose:
        print("Modelo seleccionado:", mejor, "| umbral:", umbral)
        print({k: round(v, 3) for k, v in final.items()})
        print("Matriz de confusion:", cm)
        print(importancia.head(8).round(4).to_string(index=False))
        print("Probabilidades esperadas:", esperado)
    return metricas


def _graficos(mejor, importancia, cm, probas, y_te, df):
    out = BASE / "modelo"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    top = importancia.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(top["variable"], top["importancia"], xerr=top["sd"], color=NAVY, ecolor=GOLD)
    ax.set_xlabel("Caída del ROC-AUC al permutar la variable")
    ax.set_title(f"Importancia de variables ({mejor})")
    fig.tight_layout()
    fig.savefig(out / "importancia_variables.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    cm = np.array(cm)
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1], ["No pobre", "Pobre"])
    ax.set_yticks([0, 1], ["No pobre", "Pobre"])
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title("Matriz de confusión (prueba)")
    ax.spines[["top", "right"]].set_visible(True)
    fig.tight_layout()
    fig.savefig(out / "matriz_confusion.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    colores = {"Regresion logistica (base)": "#8A8A8A", "Random Forest": GOLD, "Gradient Boosting": NAVY}
    for nombre, p in probas.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        ax.plot(fpr, tpr, lw=2, color=colores.get(nombre), label=f"{nombre} (AUC {roc_auc_score(y_te, p):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="lightgrey")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curvas ROC (prueba)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "curvas_roc.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    tasa = df.groupby("area")[OBJETIVO].mean().mul(100)
    ax.bar(tasa.index, tasa.values, color=[NAVY, GOLD][: len(tasa)])
    for i, v in enumerate(tasa.values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center")
    ax.set_ylabel("Hogares pobres (%)")
    ax.set_title("Tasa de pobreza monetaria por área")
    fig.tight_layout()
    fig.savefig(out / "pobreza_por_area.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    entrenar_y_guardar()
