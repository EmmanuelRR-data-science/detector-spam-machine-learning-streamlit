# 📩 Clasificador de Spam - Naive Bayes

Este proyecto implementa un **clasificador de mensajes como Spam o Ham** utilizando **Machine Learning** con **Multinomial Naive Bayes** y **TF-IDF**.  
Además, incluye un **dashboard interactivo en Streamlit** para probar mensajes en tiempo real y visualizar métricas y nubes de palabras.

---

## 🔗 Dashboard en vivo
Puedes probar el clasificador directamente desde tu navegador:

[**Ver el dashboard en Streamlit**](https://detector-spam-machine-learning-app-7gzuwq5muw67d8kbcixkpk.streamlit.app/)

---

## 🧩 Flujo del proyecto

1. **Carga y limpieza de datos**
   - Se carga `spam.csv` con codificación `latin-1`.
   - Se eliminan columnas innecesarias (`Unnamed: 2-4`), duplicados y valores nulos.
   - Se renombran columnas a `label` (ham/spam) y `text`.

2. **Análisis exploratorio**
   - Se visualiza la distribución de clases: `ham` vs `spam`.
   - Se identifican desbalances y características del texto.

3. **Preprocesamiento y vectorización**
   - Se utiliza **TF-IDF** para transformar texto a vectores numéricos.
   - Se consideran **unigramas y bigramas**.
   - Se limita el número máximo de features a 10,000.

4. **Entrenamiento del modelo**
   - Se usa **Multinomial Naive Bayes** con smoothing (`alpha=1.0`).
   - Se divide el dataset en **train/test split** (80/20, estratificado).

5. **Evaluación**
   - Métricas: Accuracy, Precision, Recall, F1-score.
   - Matriz de confusión para analizar errores.
   - Interpretación de palabras más asociadas a **spam** y **ham**.

6. **Interpretabilidad**
   - Se extraen las **palabras más relevantes por clase**.
   - Se visualizan con gráficas de barras y nubes de palabras.

7. **Dashboard Streamlit**
   - Permite ingresar mensajes y ver la clasificación **HAM o SPAM**.
   - Se muestra un **semaforo según confianza**:
     - 🟢 Confianza ≥ 90% → seguro
     - 🟡 Confianza 70–89% → posible spam/ham
     - 🔴 Confianza < 70% → alerta, mensaje inseguro
   - Incluye métricas interactivas y nubes de palabras para ambas clases.

---

## 📊 Ejemplos de uso en el dashboard

Mensaje: "Congratulations! You've won a free ticket to Bahamas. Call now!"
Resultado: 🔴 🚨 Mensaje muy inseguro, no entres en ningún link que contenga. (confianza: 0.56)

Mensaje: "Hey, are we still meeting at 7 for dinner?"
Resultado: 🟢 ✅ El mensaje parece muy seguro. (confianza: 0.98)

