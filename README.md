# TP3 - Análisis de sentimiento en tweets (NLP, Segundo Sprint)

Análisis de sentimiento sobre el dataset [Sentiment140](http://help.sentiment140.com/for-students) (1.6M tweets etiquetados por emoticon + 498 tweets de test etiquetados a mano). Se entrenan y evalúan dos modelos (Bag of Words + Naive Bayes, TF-IDF + Regresión Logística) siguiendo las técnicas vistas en clase, se comparan contra un modelo pre-entrenado (TextBlob), y se agrega similitud coseno como métrica de análisis.

## Estructura

```
tp3-nlp-tweets/
├── data/
│   ├── raw/            # datasets originales (ver "Cómo obtener los datos")
│   └── processed/      # texto limpio (lo genera la notebook 02, no se versiona)
├── models/             # vectorizadores y modelos entrenados (.joblib)
├── notebooks/
│   ├── 01_lectura_y_eda.ipynb     # carga + EDA
│   ├── 02_preprocesamiento.ipynb  # limpieza -> guarda los parquet
│   ├── 03_modelos.ipynb           # TextBlob + los 2 modelos -> guarda los .joblib
│   └── 04_validacion.ipynb        # pesos aprendidos + similitud coseno + conclusiones
└── requirements.txt
```

Las notebooks **se ejecutan en orden**: la 02 genera los datos limpios que consumen la 03 y la 04, y la 03 entrena los modelos que audita la 04. Cada una arranca cargando lo que necesita desde disco, así que **no hay que re-entrenar nada** para volver a mirar la validación.

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
.venv/bin/jupyter notebook notebooks/
```

Ejecutar las notebooks **en orden: 01 → 02 → 03 → 04**.

> La 02 tarda ~20s (limpia 1.6M de tweets) y la 03 ~1 min (entrena sobre los 1.6M). Las otras dos son rápidas.

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

- **Las marcas propias de Twitter sí tienen señal, y la limpieza las borra:** tener una URL sube la tasa de positivos a 66.7% (vs 49.2% sin ella), y mencionar a alguien a 59.3% (vs 42.1%) — brechas de 17 puntos. Se descartan igual porque el objetivo es comparar *técnicas de vectorización* y no mezclar efectos, pero queda documentado como la vía de mejora más concreta.
- **La hora del día también:** 59.4% de positivos a la 1 AM contra 43.3% a las 16 PM. La explicación conecta con los wordclouds — `work` era la palabra negativa más frecuente, y la gente se queja del trabajo en horario de trabajo.

Ver las notebooks para el detalle completo (cada gráfico cierra con su conclusión).
