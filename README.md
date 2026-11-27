# Estimación de la etiqueta LUNG_CANCER con aprendizaje automático

Proyecto de aula – Minería de Datos, Universidad de la Costa (prototipo académico; **no es un diagnóstico médico**).

**Integrantes:** Alejandro Serpa

**Aplicación pública:** [https://proyecto-7gzppphmsferxc3fspnpyn.streamlit.app/]

## Contenido del repositorio
- `app.py`: aplicación Streamlit (carga `model/pipeline.joblib`; no reentrena).
- `train_model.py`: ajuste, umbral, evaluación y serialización del pipeline.
- `utils.py`: limpieza y codificación compartidas por el entrenamiento y la app.
- `data/survey lung cancer.csv`: dataset (Kaggle, mysar ahmad bhat, CC0).
- `model/`: `pipeline.joblib`, `metrics.json`, `test_set_with_proba.csv`.
- `Entrega_final.ipynb`: notebook ejecutado de principio a fin (EDA, modelos, ajuste, umbral, estabilidad y prueba de consistencia).
- `requirements.txt`.

## Ejecución local
```bash
pip install -r requirements.txt
python train_model.py        # regenera el modelo y las métricas (o ejecute Entrega_final.ipynb)
streamlit run app.py
```

## Notas de versión
El modelo serializado se generó con scikit-learn 1.9.1 (fijado en `requirements.txt`). En Streamlit Community Cloud elija una versión de Python compatible con esa versión.

## Pruebas
Prueba de consistencia: tome un registro de `model/test_set_with_proba.csv`, ingréselo en la app y compare la
probabilidad con la columna `proba_YES` y con la del notebook.

## Declaración y evaluación del uso de IA
- **Herramienta y propósito:** [Claude (Anthropic), modelo/versión]. [tarea]
- **Prompt principal:** [copiar completo]
- **Respuesta recibida:** [resumen]
- **Verificación:** [qué ejecutaron y compararon]
- **Problemas identificados:** [errores, alucinaciones u omisiones; si no hubo, indíquenlo]
- **Ajustes realizados:** [qué corrigieron y qué conservaron]

Todos los integrantes comprenden el código entregado.
