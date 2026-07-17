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

## Metodología: los tres conjuntos

El dataset viene en dos archivos: `training.1600000.csv` (1.6M, etiquetado **automáticamente por emoticón**) y `testdata.manual.csv` (498, etiquetado **a mano**).

**Usarlos directamente como train y test sería un error.** Se llaman así, pero son dos conjuntos etiquetados de formas distintas: un gap train-test sólo significa algo si ambos vienen de la **misma distribución**. Comparar el F1 sobre etiquetas de emoticón contra el F1 sobre etiquetas humanas no mide sobreajuste — mide otra cosa.

| conjunto | de dónde sale | qué pregunta responde |
|---|---|---|
| **train** (1.276.996) | split 80% de los 1.6M | — |
| **test** (319.250) | split 20% de los 1.6M | **¿sobreajusta?** (misma distribución → la resta con train es válida) |
| **manual** (359) | archivo etiquetado a mano | **¿coincide con un humano?** (validación externa, se reporta aparte) |

Los vectorizadores (`CountVectorizer`, `TfidfVectorizer`) se ajustan con `fit` **solo sobre el train**; al resto se le aplica únicamente `transform`. El entrenamiento usa **todos los registros** del archivo de train, como pide la consigna.

## Resultados

**Sobreajuste** — train vs test del split (misma distribución):

| Modelo | F1 train | F1 test | Gap |
|---|---|---|---|
| BoW + Naive Bayes | 0.7910 | 0.7825 | **+0.009** |
| TF-IDF + Regresión Logística | 0.8068 | **0.8000** | **+0.007** |

**Validación externa** — set etiquetado a mano (359 tweets):

| Modelo | F1 macro |
|---|---|
| TextBlob (pre-entrenado) | 0.43 |
| BoW + Naive Bayes | **0.8273** |
| TF-IDF + Regresión Logística | 0.8215 |

*(Benchmark del azar: 0.50 — las clases están balanceadas 50/50.)*

## Conclusiones principales

- **Ningún modelo sobreajusta:** los gaps son +0.009 y +0.007, prácticamente cero. Los modelos son demasiado simples para memorizar 1.3M de ejemplos.
- **Los dos rinden MEJOR con etiquetas humanas que con las suyas propias** (0.83 vs 0.78 | 0.82 vs 0.80). El train está etiquetado por emoticón (sarcasmo, ironía, emoticones de costumbre) y el set manual a mano. → **El techo del 78-80% no es del modelo: es el ruido del etiquetado** (*weak supervision*).
- **Gana la Regresión Logística**, pero sólo se puede afirmar mirando el test interno (0.8000 vs 0.7825, sobre **319.250 tweets**). En el set manual el orden se invierte por 0.006, que sobre 359 tweets es ruido. **El tamaño del conjunto define de qué diferencias se puede hablar.**
- **Entrenar sobre el dominio le gana por lejos al pre-entrenado:** 0.82 vs 0.43 sobre el mismo set. TextBlob no puntúa mal lo negativo — **no lo ve**: su léxico es simétrico (`good`=+0.70, `bad`=−0.70), pero nadie tuitea "bad"; la gente se queja con `aching`, `ponzi`, que valen 0.00 por no estar en la lista.
- **El cuello de botella es la representación, no el algoritmo:** `'not bad'` = `not`(−5.39) + `bad`(−3.99) = negativo, cuando es un elogio. Una suma no tiene orden. Se rompe con n-gramas o BERT, no con otro clasificador.
- **El EDA anticipa el techo y detecta un problema de datos:** el largo del tweet no discrimina (70 vs 69 caracteres de mediana), y buena parte del vocabulario se comparte entre clases. La similitud coseno lo confirma: intra-clase (~0.018) vs inter-clase (0.015), apenas un 20% más. Los wordclouds además revelaron que `quot` (resto de `&quot;`) contaminaba el vocabulario → se corrigió el orden de la limpieza (`html.unescape()` **antes** de filtrar), con verificación explícita.
- **Las marcas propias de Twitter sí tienen señal, y la limpieza las borra:** una URL sube la tasa de positivos a 66.7% (vs 49.2%), y mencionar a alguien a 59.3% (vs 42.1%) — brechas de 17 puntos. Se descartan porque el objetivo es comparar *técnicas de vectorización* sin mezclar efectos, pero queda documentado como la vía de mejora más concreta.
- **La hora del día también:** 59.4% de positivos a la 1 AM contra 43.3% a las 16 PM. Conecta con los wordclouds — `work` era la palabra negativa más frecuente, y la gente se queja del trabajo en horario de trabajo.
- **La clase neutral es una limitación estructural:** no existe en el train (un tweet neutral no lleva emoticón), así que ningún modelo puede predecirla. Contra el set manual completo, su recall es 0 y la accuracy cae de 0.83 a 0.59.

Ver las notebooks para el detalle completo (cada gráfico cierra con su conclusión).
