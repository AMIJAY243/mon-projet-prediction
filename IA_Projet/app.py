import streamlit as st
import joblib
import pandas as pd

# Chargement du modèle champion
model = joblib.load('best_model.pkl')

st.title("⚡ Expert Energy Predictor")
st.write("Prédiction de la production de Charbon (Zimbabwe)")

# Entrée utilisateur : Année
annee = st.number_input("Entrez l'année (ex: 2025)", min_value=2020, max_value=2050, value=2026)

if st.button("Prédire"):
    # Création du DataFrame pour la prédiction
    input_data = pd.DataFrame([[annee]], columns=['Year'])
    
    # Prédiction
    prediction = model.predict(input_data)
    
    st.metric("Production de charbon estimée", f"{prediction[0]:.2f} unités")