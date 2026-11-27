import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from utils import FEATURES, FEATURES_BIN, LABELS_ES, TARGET, load_clean

st.set_page_config(page_title="Riesgo de cáncer de pulmón (prototipo académico)", layout="wide")

DATA_PATH = "data/survey lung cancer.csv"
AGE_MIN_FORM, AGE_MAX_FORM = 18, 100   # límites duros del formulario
AGE_MIN_DATA, AGE_MAX_DATA = 21, 87    # rango observado en el dataset (ajústelo si difiere)

INTEGRANTES = ["Alejandro Serpa"]


@st.cache_resource
def load_model():
    return joblib.load("model/pipeline.joblib")


@st.cache_data
def load_data():
    return load_clean(DATA_PATH)


@st.cache_data
def load_metrics():
    with open("model/metrics.json", encoding="utf-8") as f:
        return json.load(f)


bundle = load_model()
pipe, THR = bundle["pipeline"], bundle["threshold"]
df = load_data()
M = load_metrics()

st.title("Estimación de la etiqueta LUNG_CANCER con aprendizaje automático")
st.warning("Prototipo académico. El resultado NO es un diagnóstico médico y no sustituye "
           "la valoración de un profesional de la salud.")

tab1, tab2, tab3, tab4 = st.tabs(["Proyecto", "EDA", "Predicción", "Modelo y limitaciones"])

# ---------------------------------------------------------------- Proyecto
with tab1:
    st.header("Presentación del proyecto")
    st.markdown(
        "**Objetivo:** clasificar si un registro de una encuesta (edad, género, hábitos y síntomas) "
        "corresponde a la etiqueta YES o NO de LUNG_CANCER, considerando el fuerte desbalance de clases.\n\n"
        "**Fuente de datos:** *Lung Cancer* (Kaggle, mysar ahmad bhat, CC0) – "
        "https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer\n\n"
        f"**Datos finales:** {len(df)} registros sin duplicados y {len(FEATURES)} variables predictoras.\n\n"
        "**Asignatura:** Minería de Datos – Universidad de la Costa."
    )
    st.subheader("Integrantes")
    for n in INTEGRANTES:
        st.write("• " + n)

# --------------------------------------------------------------------- EDA
with tab2:
    st.header("Análisis exploratorio")
    c1, c2 = st.columns(2)
    with c1:
        counts = df[TARGET].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(["NO", "YES"], counts.values, color=["#4c78a8", "#e45756"])
        for i, v in enumerate(counts.values):
            ax.text(i, v, f"{v} ({v / counts.sum():.1%})", ha="center", va="bottom")
        ax.set_ylabel("Registros")
        ax.set_title("Distribución de LUNG_CANCER")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.boxplot([df[df[TARGET] == 0]["AGE"], df[df[TARGET] == 1]["AGE"]])
        ax.set_xticklabels(["NO", "YES"])
        ax.set_ylabel("Edad")
        ax.set_title("Edad según la clase")
        st.pyplot(fig)
    freq = (df.groupby(TARGET)[FEATURES_BIN].mean().T * 100).rename(index=LABELS_ES)
    fig, ax = plt.subplots(figsize=(7, 5))
    freq.plot.barh(ax=ax, color=["#4c78a8", "#e45756"])
    ax.legend(["NO", "YES"])
    ax.set_xlabel("% de registros con la característica")
    ax.set_title("Frecuencia de cada característica según la clase")
    st.pyplot(fig)
    st.caption("Las diferencias son asociaciones estadísticas en datos de encuesta, no relaciones causales.")

# ------------------------------------------------------------- Predicción
with tab3:
    st.header("Formulario de predicción")
    st.write("Responda todas las preguntas. Las variables son las mismas que usa el modelo.")
    with st.form("form"):
        c1, c2 = st.columns(2)
        gender = c1.selectbox("Género", ["Seleccione...", "Femenino", "Masculino"])
        age = c2.number_input(f"Edad (años, entre {AGE_MIN_FORM} y {AGE_MAX_FORM})",
                              min_value=AGE_MIN_FORM, max_value=AGE_MAX_FORM, value=60, step=1)
        answers = {}
        cols = st.columns(3)
        for i, f in enumerate(FEATURES_BIN):
            answers[f] = cols[i % 3].selectbox(LABELS_ES[f], ["Seleccione...", "No", "Sí"], key=f)
        submitted = st.form_submit_button("Estimar")

    if submitted:
        missing = [LABELS_ES[f] for f, v in answers.items() if v == "Seleccione..."]
        if gender == "Seleccione...":
            missing.insert(0, "Género")
        if missing:
            st.error("Faltan campos por responder: " + ", ".join(missing))
        else:
            if not (AGE_MIN_DATA <= age <= AGE_MAX_DATA):
                st.warning(f"La edad está fuera del rango observado en los datos de entrenamiento "
                           f"({AGE_MIN_DATA}–{AGE_MAX_DATA}); la estimación es menos confiable.")
            row = {"GENDER": 1 if gender == "Masculino" else 0, "AGE": int(age)}
            row.update({f: 1 if v == "Sí" else 0 for f, v in answers.items()})
            X_new = pd.DataFrame([row])[FEATURES]
            proba = float(pipe.predict_proba(X_new)[0, 1])
            label = "YES" if proba >= THR else "NO"
            st.subheader(f"Etiqueta estimada: {label}")
            st.metric("Probabilidad estimada de YES", f"{proba:.4f}")
            st.caption(f"Umbral de decisión del modelo final: {THR:.2f}. "
                       "Resultado académico; no es un diagnóstico.")

# --------------------------------------------------- Modelo y limitaciones
with tab4:
    st.header("Modelo final y métricas")
    st.write("Regresión logística con escalamiento de AGE dentro del pipeline. "
             f"Mejores parámetros (validación cruzada 5×3 sobre entrenamiento): {M['best_params']}. "
             f"Umbral de decisión: {M['test_after']['threshold']:.2f}.")

    def row(name, m):
        r = m["report"]
        return {"Conjunto": name, "Accuracy": m["accuracy"], "Balanced acc.": m["balanced_accuracy"],
                "F1 macro": m["f1_macro"], "ROC-AUC": m["roc_auc"],
                "Precision NO": r["NO"]["precision"], "Recall NO": r["NO"]["recall"],
                "Precision YES": r["YES"]["precision"], "Recall YES": r["YES"]["recall"]}

    tbl = pd.DataFrame([row("Prueba – modelo anterior (avance 2)", M["test_before"]),
                        row("Prueba – modelo final", M["test_after"]),
                        row("Entrenamiento – modelo final", M["train"])]).set_index("Conjunto")
    st.dataframe(tbl.round(3))
    st.write(f"Línea base «responder siempre YES» en prueba: accuracy = "
             f"{M['baseline_always_yes_accuracy']:.3f}.")

    cm = pd.DataFrame(M["test_after"]["confusion_matrix"],
                      index=["Real NO", "Real YES"], columns=["Pred. NO", "Pred. YES"])
    st.subheader("Matriz de confusión (conjunto de prueba, modelo final)")
    st.dataframe(cm)

    st.subheader("Limitaciones")
    st.markdown(
        "- Solo hay 38 registros de la clase NO (8 en prueba): las métricas de esa clase son inestables.\n"
        "- Datos de encuesta (casi todas las variables binarias) sin verificación clínica ni validación externa.\n"
        "- Las asociaciones observadas no son causales; el resultado no es un diagnóstico.\n"
        "- La eliminación de duplicados pudo descartar respuestas legítimas idénticas."
    )
