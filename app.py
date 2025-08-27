import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# ====================
# Cargar modelo y vectorizador
# ====================
nb = joblib.load("modelo_spam.pkl")
tfidf = joblib.load("vectorizador_tfidf.pkl")
df = pd.read_csv("spam.csv", encoding="latin-1")  # dataset limpio de spam.csv

# Verificar nombres de columnas
print("Columnas originales:", df.columns.tolist())

# Mantener solo las columnas relevantes y renombrarlas
df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})

# Limpiar nulos y duplicados
df = df.dropna().drop_duplicates()

# Verificar
print("Columnas finales:", df.columns.tolist())
print("Primeros registros:\n", df.head(3).to_string(index=False))

# ====================
# Configuración de página
# ====================
st.set_page_config(page_title="Detector de Spam", page_icon="📩", layout="wide")
st.title("📩 Detector de Spam con Naive Bayes")
st.markdown("Este modelo clasifica mensajes como **Spam** o **Ham** y muestra un semáforo según confianza.")

# ====================
# Pestañas
# ====================
tab1, tab2, tab3 = st.tabs(["🔎 Clasificador", "📊 Métricas", "☁️ Nube de Palabras"])

# ====================
# 🔎 Clasificador con semáforo
# ====================
with tab1:
    st.subheader("Prueba el clasificador")
    mensaje = st.text_area("Escribe un mensaje aquí:", height=150)

    if st.button("Clasificar"):
        if mensaje.strip() != "":
            X_new = tfidf.transform([mensaje])
            pred = nb.predict(X_new)[0]
            prob = nb.predict_proba(X_new)[0]
            conf = max(prob)  # confianza máxima del modelo

            # Semáforo según confianza
            if conf >= 0.9:
                semaforo = "🟢"
                color_text = "✅"
                texto_resultado = f"{color_text} El mensaje parece muy seguro. (confianza: {conf:.2f})"
            elif conf >= 0.7:
                semaforo = "🟡"
                color_text = "⚠️"
                texto_resultado = f"{color_text} {pred.upper()} - Posible SPAM, revisa el contenido del mensaje. (confianza: {conf:.2f})"
            else:
                semaforo = "🔴"
                color_text = "🚨"
                texto_resultado = f"{color_text} Mensaje muy inseguro, no entres en ningún link que contenga. (confianza: {conf:.2f})"

            # Mostrar resultado
            st.markdown(f"**Resultado:** {semaforo} {texto_resultado}")
        else:
            st.warning("⚠️ Por favor escribe un mensaje antes de clasificar.")

# ====================
# 📊 Métricas y matriz de confusión
# ====================
with tab2:
    st.subheader("Evaluación del modelo")

    # Vectorización del dataset completo
    X = tfidf.transform(df["text"])
    y = df["label"]
    y_pred = nb.predict(X)

    # Reporte de clasificación
    report = classification_report(y, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    st.dataframe(df_report.style.background_gradient(cmap="Blues"))

    # Matriz de confusión
    cm = confusion_matrix(y, y_pred, labels=["ham", "spam"])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["ham", "spam"], yticklabels=["ham", "spam"], ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    st.pyplot(fig)

# ====================
# ☁️ Nube de palabras
# ====================
with tab3:
    st.subheader("Nube de palabras por categoría")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### HAM")
        text_ham = " ".join(df[df["label"] == "ham"]["text"].tolist())
        wordcloud_ham = WordCloud(width=400, height=300, background_color="white").generate(text_ham)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud_ham, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    with col2:
        st.markdown("### SPAM")
        text_spam = " ".join(df[df["label"] == "spam"]["text"].tolist())
        wordcloud_spam = WordCloud(width=400, height=300, background_color="white", colormap="Reds").generate(text_spam)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud_spam, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
