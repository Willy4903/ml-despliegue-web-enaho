# Machine Learning en producción: despliegue web

Curso: Machine Learning en producción - Despliegue web. Docente: Orlando Advíncula Zeballos.

Dos proyectos con datos simulados que replican la estructura de la ENAHO (INEI):

| Proyecto | Tipo | Unidad | Variable objetivo | Modelo seleccionado |
|---|---|---|---|---|
| 1 | Regresión | Persona ocupada | `ingreso_laboral_mensual` (soles) | Gradient Boosting |
| 2 | Clasificación | Hogar | `pobre_monetario` (1 = pobre, 0 = no pobre) | Gradient Boosting |

Las bases son sintéticas: reproducen asociaciones conocidas pero no contienen personas ni hogares reales.
Para usar microdatos reales, descargar la ENAHO del portal de microdatos del INEI y reemplazar el CSV de
`data/` conservando los nombres de columnas; luego ejecutar `python entrenar.py`.

## Estructura

```
requirements.txt
.streamlit/config.toml
proyecto1_regresion/
    generar_datos.py      base simulada (semilla fija)
    entrenar.py           preprocesamiento, comparación de modelos, selección y gráficos
    modelo_utils.py       funciones compartidas por la app y por VS Code
    app.py                app Streamlit (formulario en st.sidebar)
    verificar_vscode.py   verificación de 5 casos fijos
    equidad.py            (solo Proyecto 2) desempeño por área y sexo del jefe
    data/  modelo/
proyecto2_clasificacion/  (misma estructura)
```

## Ejecución local (VS Code)

```
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

cd proyecto1_regresion
python generar_datos.py           # opcional: regenera la base
python entrenar.py                # opcional: reentrena y regenera el modelo
python verificar_vscode.py        # verificación
streamlit run app.py
```

Repetir dentro de `proyecto2_clasificacion`.

## Verificación VS Code vs Streamlit

`verificar_vscode.py` imprime la predicción de 5 casos fijos. La pestaña "Verificación" de la app
desplegada muestra los mismos 5 casos. Los valores deben ser idénticos porque ambos usan
`modelo_utils.py` y el mismo archivo `.joblib`.

## Despliegue en GitHub y Streamlit Community Cloud

1. Crear un repositorio en GitHub (por ejemplo `ml-despliegue-web-enaho`) y subir toda esta carpeta:
   ```
   git init
   git add .
   git commit -m "Proyectos 1 y 2: regresion y clasificacion"
   git branch -M main
   git remote add origin https://github.com/Willy4903/ml-despliegue-web-enaho.git
   git push -u origin main
   ```
2. Entrar a https://share.streamlit.io con la cuenta de GitHub y elegir "Create app".
3. App 1: repositorio `Willy4903/ml-despliegue-web-enaho`, rama `main`, archivo principal
   `proyecto1_regresion/app.py`. En "Advanced settings" elegir Python 3.12 y, en App URL, escribir
   `willy4903-regresion-enaho`. Deploy.
4. App 2: repetir con `proyecto2_clasificacion/app.py` y App URL `willy4903-clasificacion-enaho`.
5. Las URL finales son `https://willy4903-regresion-enaho.streamlit.app` y
   `https://willy4903-clasificacion-enaho.streamlit.app` (si un nombre estuviera ocupado, Streamlit lo avisa: usar otro y
   actualizar la diapositiva 16 del PPT).

Si la versión de scikit-learn del servidor difiere de la usada al entrenar, `cargar_modelo()` reentrena
automáticamente con la misma semilla, por lo que la app no se cae; conviene mantener
`scikit-learn==1.8.0` en `requirements.txt` para que los valores coincidan exactamente.
