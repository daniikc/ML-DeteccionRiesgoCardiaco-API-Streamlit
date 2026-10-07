"""App de Streamlit para predecir el riesgo cardíaco.

Ejecutar desde la raíz del proyecto:
    streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Añadir la raíz del proyecto al path para poder importar src
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src import config  # noqa: E402
from src.data.load_data import load_splits  # noqa: E402
from src.features.preprocessing import get_feature_types  # noqa: E402
from src.models.predict import load_model, predict_batch, predict_one  # noqa: E402

# ---------------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------------
st.set_page_config(page_title="Riesgo cardíaco", page_icon="❤️", layout="wide")

RISK_COLORS = {"Bajo": "green", "Moderado": "orange", "Alto": "red"}


# ---------------------------------------------------------------
# Carga de recursos (se ejecuta una sola vez gracias a la caché)
# ---------------------------------------------------------------
@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_train_data():
    X_train, _, _, _ = load_splits()
    return X_train


try:
    model = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

X_train = get_train_data()
num_cols, cat_cols = get_feature_types(X_train)

# ---------------------------------------------------------------
# Cabecera
# ---------------------------------------------------------------
st.title("❤️ Predicción de riesgo cardíaco")
st.caption("Proyecto educativo de Machine Learning. No sustituye una valoración médica.")

tab_one, tab_batch, tab_info = st.tabs(
    ["🔍 Un paciente", "📄 Varios pacientes (CSV)", "📊 Sobre el modelo"]
)

# ---------------------------------------------------------------
# Pestaña 1: predicción individual
# ---------------------------------------------------------------
with tab_one:
    with st.form("patient_form"):
        st.subheader("Datos del paciente")
        cols = st.columns(3)
        patient = {}

        for i, col_name in enumerate(model.feature_names_in_):
            container = cols[i % 3]

            if col_name in cat_cols:
                options = sorted(X_train[col_name].dropna().unique().tolist())
                patient[col_name] = container.selectbox(col_name, options)
            else:
                serie = X_train[col_name]
                if pd.api.types.is_integer_dtype(serie):
                    patient[col_name] = container.number_input(
                        col_name,
                        min_value=int(serie.min()),
                        max_value=int(serie.max()),
                        value=int(serie.median()),
                        step=1,
                    )
                else:
                    patient[col_name] = container.number_input(
                        col_name,
                        min_value=float(serie.min()),
                        max_value=float(serie.max()),
                        value=float(serie.median()),
                        step=0.1,
                    )

        submitted = st.form_submit_button("Calcular riesgo", type="primary")

    if submitted:
        result = predict_one(model, patient)
        color = RISK_COLORS[result["riesgo"]]

        c1, c2 = st.columns([1, 2])
        c1.metric("Probabilidad estimada", f"{result['probabilidad']:.1%}")
        c2.markdown(f"### Riesgo: :{color}[{result['riesgo']}]")
        st.progress(result["probabilidad"])

# ---------------------------------------------------------------
# Pestaña 2: predicción por lotes
# ---------------------------------------------------------------
with tab_batch:
    st.write(
        "Sube un CSV con las mismas columnas que los datos de entrenamiento. "
        "Si incluye la columna objetivo, se ignorará."
    )
    file = st.file_uploader("CSV de pacientes", type="csv")

    if file is not None:
        df = pd.read_csv(file)
        try:
            results = predict_batch(model, df)
            st.dataframe(results, width="stretch")
            st.download_button(
                "Descargar resultados",
                results.to_csv(index=False),
                file_name="predicciones.csv",
                mime="text/csv",
            )
        except ValueError as e:
            st.error(str(e))

# ---------------------------------------------------------------
# Pestaña 3: información del modelo
# ---------------------------------------------------------------
with tab_info:
    if config.METRICS_PATH.exists():
        with open(config.METRICS_PATH, encoding="utf-8") as f:
            info = json.load(f)

        st.subheader("XGBoost")
        st.caption(f"Entrenado: {info['trained_at']}")

        metrics = info["test_metrics"]
        metric_cols = st.columns(len(metrics))
        for c, (name, value) in zip(metric_cols, metrics.items()):
            c.metric(name.upper(), f"{value:.3f}")

        st.markdown("**Hiperparámetros**")
        st.json(info["params"])
    else:
        st.info("No hay métricas guardadas. Ejecuta: python -m src.models.train")