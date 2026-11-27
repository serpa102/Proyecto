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
* Herramienta y propósito: Claude (Anthropic). Se utilizó como apoyo durante el desarrollo del proyecto de estimación de la etiqueta LUNG_CANCER, principalmente para resolver dudas sobre el código y comprender algunos procedimientos relacionados con el aprendizaje automático.
* Prompt principal: Se le solicitó ayuda para revisar y comprender el código del proyecto, explicar algunos procedimientos utilizados en el entrenamiento del modelo y orientar sobre cómo mejorar su funcionamiento.
* Respuesta recibida: La herramienta proporcionó explicaciones sobre el código, recomendaciones para organizar mejor el proceso y algunas sugerencias para trabajar con los datos y evaluar los resultados del modelo.
* Verificación: Se revisaron las recomendaciones y se compararon con el funcionamiento del código en el entorno de trabajo. También se tuvo en cuenta que los resultados obtenidos fueran coherentes con el objetivo del proyecto.
* Problemas identificados: Algunas explicaciones fueron generales y necesitaron revisarse para adaptarlas al código que estábamos utilizando. También fue necesario comprobar que las sugerencias fueran compatibles con nuestro proyecto.
* Ajustes realizados: Se tomaron en cuenta las recomendaciones que ayudaban a comprender y organizar mejor el código. Se conservaron los procedimientos que ya estaban funcionando correctamente y se hicieron los cambios necesarios después de revisar su utilidad.

Todos los integrantes comprenden el código entregado y conocen el funcionamiento general del modelo desarrollado.
