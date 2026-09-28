"""
Proyecto 1 (Regresion) - App Streamlit: prediccion del ingreso laboral mensual.
Base simulada con estructura ENAHO (INEI).  Ejecutar:  streamlit run app.py
"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from generar_datos import NIVELES, SECTORES, CATEGORIAS, TAMANIOS  # noqa: F401  (listas de opciones)
from modelo_utils import (
    BASE, CASOS_VERIFICACION, COLS_CAT, COLS_NUM, FEATURES, OBJETIVO,
    cargar_datos, cargar_metricas, cargar_modelo, explicacion_local, predecir,
)

st.set_page_config(page_title="Predicción de ingreso laboral", layout="wide")
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
caso0 = CASOS_VERIFICACION[0]  # valores por defecto = caso 1 de verificacion

# ---------------------------------------------------------------- barra lateral
st.sidebar.title("Perfil de la persona ocupada")
with st.sidebar.form("formulario"):
    st.markdown("Datos personales")
    edad = st.slider("Edad (años)", 18, 65, caso0["edad"])
    sexo = st.radio("Sexo", ["Hombre", "Mujer"], index=["Hombre", "Mujer"].index(caso0["sexo"]), horizontal=True)
    area = st.selectbox("Área de residencia", ["Urbana", "Rural"], index=0)
    region = st.selectbox("Región natural", ["Lima Metropolitana", "Costa", "Sierra", "Selva"], index=0)
    miembros = st.slider("Miembros del hogar", 1, 10, caso0["miembros_hogar"])

    st.markdown("Educación y experiencia")
    nivel = st.selectbox("Nivel educativo", NIVELES, index=NIVELES.index(caso0["nivel_educativo"]))
    anios = st.slider("Años de estudio", 0, 20, caso0["anios_estudio"])
    exper = st.slider("Experiencia laboral (años)", 0, 45, caso0["experiencia_anios"])

    st.markdown("Ocupación")
    horas = st.slider("Horas trabajadas por semana", 8, 84, caso0["horas_semana"])
    categoria = st.selectbox("Categoría ocupacional", CATEGORIAS, index=CATEGORIAS.index(caso0["categoria_ocupacional"]))
    sector = st.selectbox("Sector", SECTORES, index=SECTORES.index(caso0["sector"]))
    tamanio = st.selectbox("Tamaño de la empresa", TAMANIOS, index=TAMANIOS.index(caso0["tamanio_empresa"]))
    pension = st.radio("Aporta a un sistema de pensiones", ["Si", "No"], index=0, horizontal=True)

    st.form_submit_button("Predecir ingreso", type="primary")

entrada = {
    "edad": edad, "sexo": sexo, "area": area, "region_natural": region,
    "nivel_educativo": nivel, "anios_estudio": anios, "experiencia_anios": exper,
    "horas_semana": horas, "categoria_ocupacional": categoria, "sector": sector,
    "tamanio_empresa": tamanio, "aporta_pension": pension, "miembros_hogar": miembros,
}

# ---------------------------------------------------------------- cuerpo
st.title("Predicción del ingreso laboral mensual")
st.caption(
    "Proyecto 1 - Regresión. Curso Machine Learning en producción: despliegue web. "
    "Datos simulados con la estructura de la ENAHO (INEI), módulo Empleo e Ingresos."
)

tab_pred, tab_mod, tab_datos, tab_ver = st.tabs(["Predicción", "Modelo y métricas", "Datos", "Verificación"])

with tab_pred:
    if area == "Rural" and region == "Lima Metropolitana":
        st.error("Combinación no válida: Lima Metropolitana no tiene area rural. Elija otra region o area.")
    else:
        pred = float(predecir(modelo, entrada)[0])
        mae = met["resultados"][met["modelo_seleccionado"]]["test_MAE"]
        similar = df[(df["area"] == area) & (df["nivel_educativo"] == nivel)][OBJETIVO]
        c1, c2, c3 = st.columns(3)
        c1.metric("Ingreso mensual predicho", f"S/ {pred:,.2f}")
        c2.metric("Rango orientativo (+/- MAE)", f"S/ {max(pred - mae, 0):,.0f} a {pred + mae:,.0f}")
        c3.metric(f"Mediana del grupo ({area.lower()}, {nivel.lower()})", f"S/ {similar.median():,.0f}")
        if not (0 <= anios <= 20):
            st.warning("Años de estudio fuera de rango.")

        st.subheader("Por qué se predice este ingreso")
        st.caption(
            "Efecto de cada variable: diferencia entre la predicción con el valor ingresado y la "
            "prediccion si esa variable tomara el valor tipico de la base (mediana o moda)."
        )
        exp = explicacion_local(modelo, entrada, ref).head(8).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 3.8))
        colores = [GOLD if v >= 0 else NAVY for v in exp["efecto_soles"]]
        ax.barh([f"{v} = {x}" for v, x in zip(exp["variable"], exp["valor"])], exp["efecto_soles"], color=colores)
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_xlabel("Efecto sobre el ingreso predicho (soles)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("Uso responsable de esta predicción"):
        st.markdown(
            "- Es una estimación estadística sobre datos simulados, no un salario garantizado.\n"
            "- El modelo aprende patrones de la base, incluida la brecha por sexo o área; no debe usarse para "
            "fijar remuneraciones individuales.\n"
            "- El rango mostrado (± MAE) indica la incertidumbre típica de la predicción."
        )

with tab_mod:
    sel = met["modelo_seleccionado"]
    st.subheader(f"Modelo seleccionado: {sel}")
    st.write(f"Criterio: {met['criterio']}.")
    tabla = (
        pd.DataFrame(met["resultados"]).T[["cv_RMSE", "test_MAE", "test_RMSE", "test_R2", "train_R2"]]
        .rename(columns={"cv_RMSE": "RMSE (CV)", "test_MAE": "MAE (prueba)", "test_RMSE": "RMSE (prueba)",
                         "test_R2": "R2 (prueba)", "train_R2": "R2 (entrenamiento)"})
        .round(3)
    )
    st.dataframe(tabla, width="stretch")
    a, b = st.columns(2)
    a.image(str(BASE / "modelo" / "importancia_variables.png"), caption="Importancia de variables (permutación)")
    b.image(str(BASE / "modelo" / "real_vs_predicho.png"), caption="Ingreso real vs predicho")
    st.write("Mejores hiperparámetros:", met["resultados"][sel]["mejores_parametros"])

with tab_datos:
    st.write(f"Base simulada: {df.shape[0]:,} personas ocupadas y {df.shape[1]} columnas.")
    a, b = st.columns(2)
    a.markdown("Variables numéricas")
    a.dataframe(df[COLS_NUM + [OBJETIVO]].describe().T.round(1), width="stretch")
    b.markdown("Variables categóricas")
    b.dataframe(pd.DataFrame({c: df[c].value_counts().head(6).index.tolist() + [""] * max(0, 6 - df[c].nunique())
                              for c in COLS_CAT}).T, width="stretch")
    st.image(str(BASE / "modelo" / "distribucion_ingreso.png"))
    st.dataframe(df.head(50), width="stretch")

with tab_ver:
    st.write(
        "Cinco casos fijos. Ejecute `python verificar_vscode.py` en VS Code: los montos deben ser "
        "idénticos a los de esta tabla."
    )
    pred_app = predecir(modelo, CASOS_VERIFICACION)
    ver = pd.DataFrame(
        {
            "Caso": [c["nombre"] for c in CASOS_VERIFICACION],
            "Predicción app (S/)": pred_app,
            "Esperado del entrenamiento (S/)": met["casos_verificacion_esperados"],
        }
    )
    ver["Coincide"] = (ver.iloc[:, 1] - ver.iloc[:, 2]).abs() < 0.01
    st.dataframe(ver, width="stretch", hide_index=True)
