"""Entrena el pipeline final, elige el umbral SOLO con datos de entrenamiento,
evalúa una única vez en prueba y guarda todo para la app.

Uso:  python train_model.py
Salidas: model/pipeline.joblib, model/metrics.json, model/test_set_with_proba.csv

Copie esta misma lógica al notebook (o ejecute este script desde él) para que
las cifras del artículo salgan del notebook.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import FEATURES, TARGET, load_clean

DATA_PATH = "data/survey lung cancer.csv"
SEED = 42
MIN_RECALL_YES = 0.90  # restricción para elegir el umbral (ajuste según el costo del error)

os.makedirs("model", exist_ok=True)

df = load_clean(DATA_PATH)
X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"Registros: {len(df)} | train {len(X_train)} | test {len(X_test)}")
print("Distribución:", y.value_counts().to_dict())


def make_pipe():
    prep = ColumnTransformer([("age", StandardScaler(), ["AGE"])], remainder="passthrough")
    return Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000))])


def evaluate(y_true, proba, thr):
    pred = (proba >= thr).astype(int)
    return {
        "threshold": float(thr),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "f1_macro": f1_score(y_true, pred, average="macro"),
        "roc_auc": roc_auc_score(y_true, proba),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),  # filas real NO/YES
        "report": classification_report(y_true, pred, target_names=["NO", "YES"],
                                        output_dict=True, zero_division=0),
    }


# 1) Modelo anterior (avance 2): regresión logística por defecto, umbral 0.5
base = make_pipe().fit(X_train, y_train)
metrics_before = evaluate(y_test, base.predict_proba(X_test)[:, 1], 0.5)

# 2) Ajuste de hiperparámetros con validación cruzada repetida (solo entrenamiento)
grid = {"clf__C": [0.01, 0.1, 1, 10, 100], "clf__class_weight": [None, "balanced"]}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
gs = GridSearchCV(make_pipe(), grid, scoring="balanced_accuracy", cv=cv).fit(X_train, y_train)
best = gs.best_estimator_
cv_mean = gs.best_score_
cv_std = gs.cv_results_["std_test_score"][gs.best_index_]
print("Mejores parámetros:", gs.best_params_, f"| balanced acc CV {cv_mean:.3f} ± {cv_std:.3f}")

# 3) Umbral elegido solo con predicciones de validación cruzada sobre entrenamiento
proba_cv = cross_val_predict(
    best, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    method="predict_proba")[:, 1]
rows = []
for t in np.arange(0.30, 0.951, 0.05):
    pred = (proba_cv >= t).astype(int)
    rec_yes = ((pred == 1) & (y_train == 1)).sum() / (y_train == 1).sum()
    rec_no = ((pred == 0) & (y_train == 0)).sum() / (y_train == 0).sum()
    rows.append({"umbral": round(float(t), 2), "recall_YES": rec_yes, "recall_NO": rec_no})
thr_table = pd.DataFrame(rows)
print(thr_table.round(3).to_string(index=False))
ok = thr_table[thr_table["recall_YES"] >= MIN_RECALL_YES]
threshold = float(ok.sort_values(["recall_NO", "umbral"], ascending=[False, True]).iloc[0]["umbral"]) \
    if len(ok) else 0.5
print("Umbral elegido:", threshold)

# 3b) Evaluación final UNA sola vez en prueba
proba_test = best.predict_proba(X_test)[:, 1]
metrics_after = evaluate(y_test, proba_test, threshold)
metrics_train = evaluate(y_train, best.predict_proba(X_train)[:, 1], threshold)

print("\nANTES  (avance 2):", {k: round(v, 3) for k, v in metrics_before.items()
                               if isinstance(v, float)})
print("DESPUÉS (final)  :", {k: round(v, 3) for k, v in metrics_after.items()
                           if isinstance(v, float)})
print("Matriz de confusión final (filas real NO/YES):", metrics_after["confusion_matrix"])

# 4) Guardar pipeline completo + metadatos
joblib.dump({"pipeline": best, "threshold": threshold, "features": FEATURES},
            "model/pipeline.joblib")

out = {
    "best_params": {k: (None if v is None else v) for k, v in gs.best_params_.items()},
    "cv_balanced_accuracy_mean": float(cv_mean),
    "cv_balanced_accuracy_std": float(cv_std),
    "threshold_table": thr_table.round(4).to_dict(orient="records"),
    "min_recall_yes": MIN_RECALL_YES,
    "baseline_always_yes_accuracy": float(y_test.mean()),
    "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    "train": metrics_train, "test_before": metrics_before, "test_after": metrics_after,
}
with open("model/metrics.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=float)

# Conjunto de prueba con la probabilidad del modelo final: sirve para la prueba
# de consistencia notebook vs. Streamlit (mismo registro, misma probabilidad).
test_out = X_test.copy()
test_out["y_real"] = y_test
test_out["proba_YES"] = proba_test
test_out.to_csv("model/test_set_with_proba.csv", index=True)
print("Guardado en model/. Use test_set_with_proba.csv para la prueba de consistencia.")
