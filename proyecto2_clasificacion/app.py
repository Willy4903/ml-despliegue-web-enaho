"""
Proyecto 2 (Clasificacion) - App Streamlit: riesgo de pobreza monetaria de un hogar.
Base simulada con estructura ENAHO (INEI).  Ejecutar:  streamlit run app.py
"""
import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from generar_datos import NIVELES, OCUPACIONES, PISOS, TENENCIAS
from modelo_utils import (
    BASE, CASOS_VERIFICACION, COLS_CAT, COLS_NUM, OBJETIVO, cargar_datos, cargar_metricas,
    cargar_modelo, clasificar, explicacion_local, probabilidad,
)

st.set_page_config(page_title="Riesgo de pobreza monetaria", layout="wide")
NAVY, GOLD = "#0D3B66", "#E8A537"


@st.cache_resource
def modelo_cache():
    return cargar_modelo()


@st.cache_data
def datos_cache():
    return cargar_datos()


modelo = modelo_cache()
df = datos_cache()
met = cargar_metricas()
ref = met["referencia"]
c0 = CASOS_VERIFICACION[0]  # valores por defecto = caso 1 de verificacion

# ---------------------------------------------------------------- barra lateral
st.sidebar.title("Características del hogar")
with st.sidebar.form("formulario"):
    st.markdown("Ubicación")
    area = st.selectbox("Área de residencia", ["Urbana", "Rural"], index=0)
    region = st.selectbox("Región natural", ["Lima Metropolitana", "Costa", "Sierra", "Selva"], index=0)

    st.markdown("Jefe del hogar")
    edad = st.slider("Edad del jefe (años)", 22, 85, c0["edad_jefe"])
    sexo = st.radio("Sexo del jefe", ["Hombre", "Mujer"], index=0, horizontal=True)
    nivel = st.selectbox("Nivel educativo del jefe", NIVELES, index=NIVELES.index(c0["nivel_educativo_jefe"]))
    ocup = st.selectbox("Situación laboral del jefe", OCUPACIONES, index=OCUPACIONES.index(c0["ocupacion_jefe"]))

    st.markdown("Composición del hogar")
    tam = st.slider("Miembros del hogar", 1, 10, c0["tamanio_hogar"])
    menores = st.slider("Menores de 14 años", 0, 6, c0["menores_14"])
    perc = st.slider("Perceptores de ingreso", 0, 5, c0["perceptores_ingreso"])

    st.markdown("Vivienda y servicios")
    hab = st.slider("Habitaciones", 1, 8, c0["habitaciones"])
    piso = st.selectbox("Material del piso", PISOS, index=PISOS.index(c0["material_piso"]))
    tenencia = st.selectbox("Tenencia de la vivienda", TENENCIAS, index=TENENCIAS.index(c0["tenencia_vivienda"]))
    agua = st.radio("Agua por red pública", ["Si", "No"], index=0, horizontal=True)
    elec = st.radio("Electricidad", ["Si", "No"], index=0, horizontal=True)
    internet = st.radio("Internet en el hogar", ["Si", "No"], index=0, horizontal=True)

    st.markdown("Decisión del modelo")
    umbral = st.slider(
        "Umbral de clasificación", 0.10, 0.90, float(met["umbral_optimo"]), 0.01,
        help="Probabilidad mínima para clasificar el hogar como pobre. El valor por defecto maximiza el F1 en validación.",
    )
    st.form_submit_button("Evaluar hogar", type="primary")

entrada = {
    "area": area, "region_natural": region, "edad_jefe": edad, "sexo_jefe": sexo,
    "nivel_educativo_jefe": nivel, "ocupacion_jefe": ocup, "tamanio_hogar": tam,
    "menores_14": menores, "perceptores_ingreso": perc, "habitaciones": hab,
    "agua_red_publica": agua, "electricidad": elec, "internet": internet,
    "material_piso": piso, "tenencia_vivienda": tenencia,
}

# ---------------------------------------------------------------- cuerpo
st.title("Riesgo de pobreza monetaria del hogar")
st.caption(
    "Proyecto 2 - Clasificación. Curso Machine Learning en producción: despliegue web. "
    "Datos simulados con la estructura de la ENAHO (INEI). Variable objetivo: 1 = pobre, 0 = no pobre."
)

tab_pred, tab_mod, tab_datos, tab_ver = st.tabs(["Predicción", "Modelo y métricas", "Datos", "Verificación"])

with tab_pred:
    if area == "Rural" and region == "Lima Metropolitana":
        st.error("Combinación no válida: Lima Metropolitana no tiene área rural. Elija otra región o área.")
    elif menores >= tam:
        st.error("Los menores de 14 años no pueden igualar o superar el total de miembros del hogar.")
    else:
        p = float(probabilidad(modelo, entrada)[0])
        clase = int(clasificar(p, umbral))
        c1, c2, c3 = st.columns(3)
        c1.metric("Probabilidad de pobreza", f"{p:.1%}")
        c2.metric("Clase predicha", "1 (Pobre)" if clase == 1 else "0 (No pobre)")
        c3.metric("Tasa de pobreza en el área", f"{df[df['area'] == area][OBJETIVO].mean():.1%}")
        st.progress(min(max(p, 0.0), 1.0), text=f"Probabilidad {p:.1%} frente al umbral {umbral:.0%}")

        st.subheader("Qué eleva o reduce el riesgo")
        st.caption(
            "Cambio en la probabilidad al reemplazar cada variable por el valor típico de la base "
            "(mediana o moda). Barras doradas elevan el riesgo; barras azules lo reducen."
        )
        exp = explicacion_local(modelo, entrada, ref).head(8).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 3.8))
        colores = [GOLD if v >= 0 else NAVY for v in exp["efecto_prob"]]
        ax.barh([f"{v} = {x}" for v, x in zip(exp["variable"], exp["valor"])], exp["efecto_prob"] * 100, color=colores)
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_xlabel("Efecto sobre la probabilidad (puntos porcentuales)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("Uso responsable de esta predicción"):
        st.markdown(
            "- Es una estimación estadística sobre datos simulados, no un diagnóstico del hogar.\n"
            "- Los falsos negativos dejan sin apoyo a hogares pobres; los falsos positivos gastan recursos. "
            "El umbral decide ese equilibrio.\n"
            "- El desempeño difiere entre grupos (ver pestaña Modelo y métricas). Una persona debe revisar "
            "las decisiones que afecten a hogares reales."
        )

with tab_mod:
    sel = met["modelo_seleccionado"]
    st.subheader(f"Modelo seleccionado: {sel}")
    st.write(f"Criterio: {met['criterio']}. Umbral óptimo (F1): {met['umbral_optimo']:.2f}.")
    tabla = (
        pd.DataFrame(met["resultados"]).T[
            ["cv_ROC_AUC", "test_ROC_AUC", "test_PR_AUC", "test_Accuracy", "test_Precision", "test_Recall", "test_F1"]
        ]
        .rename(columns=lambda c: c.replace("test_", "").replace("cv_", "CV ").replace("_", "-"))
        .round(3)
    )
    st.dataframe(tabla, width="stretch")
    st.caption("Las métricas de la tabla usan umbral 0,50 para todos los modelos, por comparabilidad.")
    f = met["metricas_finales_prueba"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC (prueba)", f"{f['ROC_AUC']:.3f}")
    m2.metric("Recall al umbral óptimo", f"{f['Recall']:.3f}")
    m3.metric("Precisión al umbral óptimo", f"{f['Precision']:.3f}")
    m4.metric("F1 al umbral óptimo", f"{f['F1']:.3f}")
    a, b, c = st.columns(3)
    a.image(str(BASE / "modelo" / "importancia_variables.png"), caption="Importancia de variables (permutación)")
    b.image(str(BASE / "modelo" / "matriz_confusion.png"), caption="Matriz de confusión al umbral óptimo")
    c.image(str(BASE / "modelo" / "curvas_roc.png"), caption="Curvas ROC")
    st.write("Mejores hiperparámetros:", met["resultados"][sel]["mejores_parametros"])

    st.subheader("Revisión de equidad por grupo")
    ruta_eq = BASE / "modelo" / "equidad_grupos.json"
    if ruta_eq.exists():
        eq = pd.DataFrame(json.loads(ruta_eq.read_text(encoding="utf-8"))).T
        st.dataframe(eq, width="stretch")
        st.caption(
            "Métricas en el conjunto de prueba con el umbral óptimo. El modelo detecta bastante mejor a los "
            "hogares pobres rurales que a los urbanos: conviene revisar el umbral por grupo antes de usarlo "
            "para focalizar recursos."
        )

with tab_datos:
    st.write(f"Base simulada: {df.shape[0]:,} hogares y {df.shape[1]} columnas. Prevalencia de pobreza: {df[OBJETIVO].mean():.1%}.")
    a, b = st.columns(2)
    a.markdown("Variables numéricas")
    a.dataframe(df[COLS_NUM].describe().T.round(1), width="stretch")
    b.markdown("Variables categóricas (categorías más frecuentes)")
    b.dataframe(
        pd.DataFrame({"categorías": [", ".join(df[c].value_counts().head(3).index.astype(str)) for c in COLS_CAT]},
                     index=COLS_CAT),
        width="stretch",
    )
    st.image(str(BASE / "modelo" / "pobreza_por_area.png"))
    st.dataframe(df.head(50), width="stretch")

with tab_ver:
    st.write(
        "Cinco casos fijos. Ejecute `python verificar_vscode.py` en VS Code: las probabilidades deben ser "
        "idénticas a las de esta tabla."
    )
    prob_app = probabilidad(modelo, CASOS_VERIFICACION)
    ver = pd.DataFrame(
        {
            "Caso": [c["nombre"] for c in CASOS_VERIFICACION],
            "Probabilidad app": prob_app,
            "Esperada del entrenamiento": met["casos_verificacion_esperados"],
            "Clase (umbral óptimo)": clasificar(prob_app, met["umbral_optimo"]),
        }
    )
    ver["Coincide"] = (ver["Probabilidad app"] - ver["Esperada del entrenamiento"]).abs() < 1e-4
    st.dataframe(ver, width="stretch", hide_index=True)
