import streamlit as st
import joblib
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Configuration pour trouver les fichiers automatiquement
base_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Fonction pour charger les assets (modèles et données)
@st.cache_resource
def load_assets():
    model = joblib.load(os.path.join(base_dir, 'model.pkl'))
    scaler = joblib.load(os.path.join(base_dir, 'scaler.pkl'))
    df = pd.read_csv(os.path.join(base_dir, 'energy_data.csv'))
    return model, scaler, df

# 3. Chargement des données
model, scaler, df = load_assets()

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="Expert Energy Predictor", page_icon="⚡", layout="wide")

# --- DESIGN MODERNE (CSS) ---
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #1e3a8a !important; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    .stApp { background: linear-gradient(135deg, #fff5e6 0%, #ffe0b3 100%); }
    .stApp, .stApp h1, .stApp p, .stApp div, .stApp label { color: #000000 !important; }
    h1 { text-align: center; background: rgba(255, 255, 255, 0.7); padding: 20px; border-radius: 15px; }
    div.stButton > button:first-child { background-color: #f59e0b !important; color: #000000 !important; font-weight: bold !important; }
    [data-testid="stMetric"] { background-color: rgba(255, 255, 255, 0.9); padding: 20px; border-radius: 15px; border-left: 5px solid #f59e0b; }
    </style>
    """, unsafe_allow_html=True)

# Préparation données
df_zimbabwe = df[df['Entity'] == 'Zimbabwe'].dropna(subset=['Coal'])

# --- SIDEBAR ---
st.sidebar.header("⚙️ Contrôles")
annee = st.sidebar.slider("Année de prédiction :", 2026, 2050, 2030)

# --- UI PRINCIPALE ---
st.title("⚡ Expert Energy Predictor")

tab1, tab2, tab3, tab4 = st.tabs(["🚀 Simulation", "📈 Analyse Graphique", "🎯 Objectifs", "👤 Identifiants"])

with tab1:
    st.subheader("Simulation de production")
    if st.button("Lancer la prédiction", key="btn_pred"):
        input_data = pd.DataFrame([[annee]], columns=['Year'])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        st.metric(label=f"Production estimée {annee}", value=f"{prediction:.2f} GWh")
        st.session_state.last_pred = prediction
        st.session_state.last_year = annee

with tab2:
    st.subheader("Analyse Graphique")
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.lineplot(data=df_zimbabwe, x='Year', y='Coal', marker='o', color='#1e3a8a', linewidth=2.5, ax=ax, label='Historique')
    
    if 'last_pred' in st.session_state:
        ax.plot(st.session_state.last_year, st.session_state.last_pred, marker='*', markersize=18, color='#f59e0b', linestyle='None', label='Prédiction')
        last_real = df_zimbabwe.iloc[-1]
        ax.plot([last_real['Year'], st.session_state.last_year], [last_real['Coal'], st.session_state.last_pred], linestyle='--', color='#f59e0b', alpha=0.6)

    ax.set_xlim(df_zimbabwe['Year'].min(), 2051)
    plt.xticks(rotation=45)
    ax.set_title("Évolution et Projection (Zimbabwe)")
    st.pyplot(fig)

with tab3:
    st.subheader("Objectifs du projet")
    c1, c2, c3 = st.columns(3)
    c1.info("🔍 Analyse des tendances")
    c2.info("🔮 Prévision par modèle SVR")
    c3.info("⚖️ Aide à l'investissement")

with tab4:
    st.subheader("👤 Identifiants du Développeur")
    nom = st.text_input("Nom complet :")
    tel = st.text_input("Téléphone :")
    mail = st.text_input("Adresse Gmail :")
    if st.button("Valider"):
        if nom and tel and "@gmail.com" in mail:
            st.success(f"Enregistré pour {nom} !")
        else:
            st.warning("Veuillez remplir correctement les champs.")
