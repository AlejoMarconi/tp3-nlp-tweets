# TP3 — Análisis de sentimiento en tweets (Sentiment140)

Trabajo práctico de NLP de la Diplomatura en Data Science / Machine Learning.
Se entrena y evalúa un clasificador binario de sentimiento (positivo / negativo) sobre el dataset **Sentiment140** (~1.6 millones de tweets), y se compara su rendimiento contra un modelo pre-entrenado (**TextBlob**), según la opción 1 de la consigna.

Todo el desarrollo está en [notebook.ipynb](notebook.ipynb).

## Dataset

[Sentiment140](http://help.sentiment140.com/for-students): tweets en inglés recolectados en 2009, etiquetados automáticamente según los emoticones que contenían.

- `training.1600000.processed.noemoticon.csv` — 1.6M de tweets etiquetados (`0` = negativo, `4` = positivo). Es el archivo que usa el notebook, completo, para entrenar y evaluar.
- `testdata.manual.2009.06.14.csv` — conjunto de test etiquetado a mano, incluido en el repo como referencia (no se usa en el análisis).

Columnas: `sentiment`, `tweet_id`, `date`, `query`, `user`, `text`.

## Contenido del notebook

1. **Carga y preparación de datos** — revisión de nulos, duplicados por ID y por texto, y eliminación de tweets con etiquetas contradictorias (mismo texto etiquetado como positivo y negativo a la vez).
2. **EDA** — variables derivadas del texto (longitud, cantidad de palabras, hashtags, menciones, URLs, exclamaciones, preguntas, uso de mayúsculas) y variables temporales (hora, día de la semana, franja horaria, momento del mes), comparadas entre clases. También se revisa una muestra de etiquetas para dimensionar el ruido del etiquetado automático.
3. **Preprocesamiento** — minúsculas, decodificación de entidades HTML, reemplazo de URLs y menciones por tokens genéricos (`url`, `user`), reemplazo de exclamaciones y preguntas por tokens (`exclamacion`, `pregunta`), conservación del contenido de los hashtags y limpieza de caracteres especiales.
4. **Representación** — TF-IDF con unigramas y bigramas (vocabulario máximo de 50.000 términos, `min_df=5`, `max_df=0.95`, `sublinear_tf`).
5. **Modelo** — Regresión Logística (solver `saga`), con split estratificado 80/20 y semilla fija.
6. **Evaluación** — accuracy, precision, recall, F1 por clase y F1 macro; matriz de confusión; comparación entrenamiento vs. prueba para chequear sobreajuste.
7. **Interpretación** — términos con mayor peso positivo y negativo en el modelo, incluyendo el efecto de los tokens especiales conservados en la limpieza.
8. **Análisis de errores** — revisión de falsos positivos y falsos negativos (predominan tweets con sentimientos mixtos o dependientes del contexto).
9. **Comparación con TextBlob** — clasificación por polaridad del léxico sobre el mismo conjunto de prueba.

## Resultados

| Modelo | Accuracy | F1 macro |
|---|---|---|
| TF-IDF + Regresión Logística | 82.4 % | 82.4 % |
| TextBlob (pre-entrenado) | 62.6 % | 62.4 % |

- El modelo entrenado se comporta de forma equilibrada entre ambas clases (precision, recall y F1 prácticamente iguales para positivos y negativos).
- La diferencia de F1 macro entre entrenamiento (83.6 %) y prueba (82.3 %) es de ~1.25 puntos, sin señales importantes de overfitting.
- TextBlob rinde bastante peor: usa un léxico general, no está entrenado con tweets, y asignó polaridad neutra (0) a más de 112.000 tweets del conjunto de prueba, que en la binarización quedan como negativos.

## Cómo ejecutarlo

Requiere Python 3.11 y los dos CSV de Sentiment140 en la raíz del repositorio (el de entrenamiento pesa ~230 MB y no está versionado; se descarga del [sitio de Sentiment140](http://help.sentiment140.com/for-students)).

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn seaborn matplotlib textblob jupyter
jupyter notebook notebook.ipynb
```

Nota: el entrenamiento y la inferencia de TextBlob sobre los 1.6M de registros llevan varios minutos.
