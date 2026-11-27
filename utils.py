"""Limpieza y codificación compartidas por train_model.py y app.py.

Usar la misma función en ambos sitios evita diferencias entre notebook y app.
"""
import pandas as pd

TARGET = "LUNG_CANCER"
FEATURES_BIN = [
    "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", "CHRONIC_DISEASE",
    "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL_CONSUMING", "COUGHING",
    "SHORTNESS_OF_BREATH", "SWALLOWING_DIFFICULTY", "CHEST_PAIN",
]
FEATURES = ["GENDER", "AGE"] + FEATURES_BIN  # orden que espera el pipeline

LABELS_ES = {
    "SMOKING": "Tabaquismo",
    "YELLOW_FINGERS": "Dedos amarillos",
    "ANXIETY": "Ansiedad",
    "PEER_PRESSURE": "Presión de grupo",
    "CHRONIC_DISEASE": "Enfermedad crónica",
    "FATIGUE": "Fatiga",
    "ALLERGY": "Alergia",
    "WHEEZING": "Sibilancias",
    "ALCOHOL_CONSUMING": "Consumo de alcohol",
    "COUGHING": "Tos",
    "SHORTNESS_OF_BREATH": "Dificultad para respirar",
    "SWALLOWING_DIFFICULTY": "Dificultad para tragar",
    "CHEST_PAIN": "Dolor de pecho",
}


def load_clean(path):
    """Carga el CSV original de Kaggle y aplica la limpieza del proyecto.

    - normaliza nombres de columnas (espacios -> guion bajo)
    - elimina duplicados
    - GENDER: M = 1, F = 0; binarias 1/2 -> 0/1; objetivo YES = 1, NO = 0
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    df = df.drop_duplicates().reset_index(drop=True)
    df["GENDER"] = (df["GENDER"].astype(str).str.strip() == "M").astype(int)
    for c in FEATURES_BIN:
        df[c] = df[c].map({1: 0, 2: 1})
    df[TARGET] = (df[TARGET].astype(str).str.strip() == "YES").astype(int)
    assert df[FEATURES + [TARGET]].notna().all().all(), "Hay valores sin codificar"
    return df
