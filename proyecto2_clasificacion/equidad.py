"""
Proyecto 2 - Revisión de equidad: desempeño del modelo por grupo (área y sexo del jefe)
en el conjunto de prueba, con el umbral óptimo. Guarda modelo/equidad_grupos.json.

Ejecutar:  python equidad.py
"""
import json

import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

from modelo_utils import BASE, FEATURES, OBJETIVO, cargar_datos, cargar_metricas, cargar_modelo


def calcular() -> dict:
    df, modelo, met = cargar_datos(), cargar_modelo(), cargar_metricas()
    X, y = df[FEATURES], df[OBJETIVO]
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.20, random_state=met["semilla"], stratify=y)
    pred = pd.Series((modelo.predict_proba(X_te)[:, 1] >= met["umbral_optimo"]).astype(int), index=X_te.index)
    out = {}
    for col in ["area", "sexo_jefe"]:
        for grupo, idx in X_te.groupby(col).groups.items():
            yy, pp = y_te.loc[idx], pred.loc[idx]
            out[f"{col}={grupo}"] = {
                "hogares": int(len(idx)),
                "pobres_reales": int(yy.sum()),
                "recall": round(float(recall_score(yy, pp)), 3),
                "precision": round(float(precision_score(yy, pp)), 3),
                "tasa_real": round(float(yy.mean()), 3),
                "tasa_predicha": round(float(pp.mean()), 3),
            }
    return out


if __name__ == "__main__":
    res = calcular()
    with open(BASE / "modelo" / "equidad_grupos.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(pd.DataFrame(res).T.to_string())
