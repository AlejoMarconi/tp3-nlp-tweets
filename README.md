# TP3 - Análisis de sentimiento en tweets (NLP, Segundo Sprint)

Análisis de sentimiento sobre el dataset [Sentiment140](http://help.sentiment140.com/for-students) (1.6M tweets etiquetados por emoticon + 498 tweets de test etiquetados a mano). Se entrenan y evalúan dos modelos (Bag of Words + Naive Bayes, TF-IDF + Regresión Logística) siguiendo las técnicas vistas en clase, se comparan contra un modelo pre-entrenado (TextBlob), y se agrega similitud coseno como métrica de análisis.

## Estructura

```
tp3-nlp-tweets/
├── data/raw/           # dataset (ver "Cómo obtener los datos" abajo)
├── notebooks/
│   └── nlp_sentiment_analysis.ipynb   # notebook único con todo el análisis
└── requirements.txt
```

## Cómo obtener los datos

El archivo de entrenamiento (`training.1600000.processed.noemoticon.csv`, ~230MB) no está versionado por su tamaño. Descargar de Google Drive y descomprimir en `data/raw/`:

```bash
uvx gdown "https://drive.google.com/uc?id=0B04GJPshIjmPRnZManQwWEdTZjg" -O data/raw/sentiment140.zip
unzip data/raw/sentiment140.zip -d data/raw/
```

El archivo de test manual (`testdata.manual.2009.06.14.csv`, 74KB) sí está incluido.

## Cómo correr

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
.venv/bin/python3 -m spacy download en_core_web_sm
.venv/bin/python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
.venv/bin/jupyter notebook notebooks/nlp_sentiment_analysis.ipynb
```

## Resultados principales

Todos los modelos se evalúan en **ambos conjuntos** (train y test) para poder detectar sobreajuste. Los vectorizadores (`CountVectorizer`, `TfidfVectorizer`) se ajustan con `fit` **solo sobre el train**; al test se le aplica únicamente `transform`. El entrenamiento usa los **1.6M de registros completos**, como pide la consigna.

| Modelo | F1 macro (train) | F1 macro (test) | Accuracy (test) | Gap |
|---|---|---|---|---|
| TextBlob (pre-entrenado) | — | 0.43 | 0.59 | — |
| BoW + Naive Bayes | 0.7907 | 0.8189 | 0.82 | −0.028 |
| TF-IDF + Regresión Logística | 0.8060 | **0.8245** | 0.82 | −0.018 |

*(Benchmark del azar: 0.50 — el train está balanceado 50/50: 800.000 positivos y 800.000 negativos.)*

## Conclusiones principales

- **Entrenar sobre el dominio le gana a un pre-entrenado genérico:** 0.82 vs 0.59 de accuracy. TextBlob usa un léxico de texto formal y no reconoce el lenguaje de Twitter (su recall sobre negativos cae a 0.42).
- **Ningún modelo sobreajusta: los gaps son negativos**, el test rinde *mejor* que el train. Parece contraintuitivo pero es lo esperable: el train está etiquetado **automáticamente por emoticón** (ruidoso — sarcasmo, sentimientos mezclados) y el test **a mano por humanos** (limpio). El ~79-81% de train no es el techo del modelo sino el techo que impone el ruido de las etiquetas (*weak supervision*).
- **Los dos modelos entrenados empatan en la práctica (~0.82):** la diferencia de 0.006 sobre un test de 359 tweets son ~2 casos, no alcanza para declarar un ganador. El cuello de botella es la representación, no el algoritmo: ambos son lineales sobre bag-of-words y ninguno captura orden de palabras, negación ni contexto. Para superar ese techo habría que cambiar la representación (n-gramas, BERT) o mejorar las etiquetas.
- **El EDA anticipa el techo y detecta un problema de datos:** el largo del tweet no discrimina sentimiento (70 vs 69 caracteres de mediana), y buena parte del vocabulario se comparte entre clases. La similitud coseno lo confirma: la intra-clase (~0.018) supera a la inter-clase (0.015) apenas un 20%. Los wordclouds además revelaron que `quot` (resto de la entidad HTML `&quot;`) contaminaba el vocabulario, lo que llevó a corregir el orden de la limpieza: `html.unescape()` **antes** de filtrar caracteres no alfabéticos, con verificación explícita en la notebook.
- **La clase neutral es una limitación estructural del dataset:** no existe en el train (un tweet neutral no lleva emoticón), así que ningún modelo puede predecirla. Contra el test completo, su recall es 0 y la accuracy cae a 0.59.

Ver la notebook para el detalle completo (EDA con conclusiones, wordclouds, análisis de sensibilidad de TextBlob, similitud coseno).
