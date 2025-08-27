import streamlit as st
import joblib

# Cargar el modelo y el vectorizador
nb = joblib.load("modelo_spam.pkl")
tfidf = joblib.load("vectorizador_tfidf.pkl")

# Interfaz
st.set_page_config(page_title="Detector de Spam", page_icon="📩")

st.title("📩 Detector de Spam con Naive Bayes")
st.write("Escribe un mensaje y el modelo te dirá si es **Spam** o **Ham**.")

# Entrada del usuario
mensaje = st.text_area("Escribe el mensaje aquí:", height=150)

if st.button("Clasificar"):
    if mensaje.strip() != "":
        X_new = tfidf.transform([mensaje])
        pred = nb.predict(X_new)[0]
        prob = nb.predict_proba(X_new)[0]

        st.subheader("Resultado:")
        if pred == "spam":
            st.error(f"🚨 SPAM (confianza: {max(prob):.2f})")
        else:
            st.success(f"✅ HAM (confianza: {max(prob):.2f})")
    else:
        st.warning("Por favor escribe un mensaje antes de clasificar.")
