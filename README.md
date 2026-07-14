# TP3 - Análisis de sentimiento en tweets (NLP, Segundo Sprint)

Análisis de sentimiento sobre el dataset [Sentiment140](http://help.sentiment140.com/for-students) (1.6M tweets etiquetados por emoticon + 498 tweets de test etiquetados a mano). Se entrenan y evalúan dos modelos (Bag of Words + Naive Bayes, TF-IDF + Regresión Logística) siguiendo las técnicas vistas en clase, se comparan contra un modelo pre-entrenado (TextBlob), se agrega similitud coseno como métrica de análisis, y como opcional se entrenan embeddings Word2Vec propios con un juego de analogías.

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

| Modelo | Accuracy | F1 (macro) |
|---|---|---|
| TextBlob (pre-entrenado) | ~0.59 | ~0.43 |
| BoW + Naive Bayes | ~0.82 | ~0.82 |
| TF-IDF + Regresión Logística | ~0.82 | ~0.82 |

Ver la notebook para el detalle completo (EDA, wordclouds, similitud coseno, embeddings y analogías, conclusiones).
