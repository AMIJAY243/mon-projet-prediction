import seaborn as sns
import streamlit as st
import joblib
import pandas as pd
import os

# 1. Configuration pour trouver les fichiers automatiquement
base_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Fonction pour charger les assets (modèles et données)
@st.cache_resource
def load_assets():
    # On utilise os.path.join pour trouver les fichiers dans le même dossier que app.py
    model = joblib.load(os.path.join(base_dir, 'model.pkl'))
    scaler = joblib.load(os.path.join(base_dir, 'scaler.pkl'))
    df = pd.read_csv(os.path.join(base_dir, 'energy_data.csv'))
    return model, scaler, df

# 3. Chargement
model, scaler, df = load_assets()

# --- ICI TU COLLES LE RESTE DE TON CODE D'INTERFACE ---
# Assure-toi que tu n'as plus aucun chemin commençant par "C:\" dans ton code
# --- CONFIGURATION ---
st.set_page_config(page_title="Expert Energy Predictor", page_icon="⚡", layout="wide")

# --- DESIGN MODERNE (CSS) ---
st.markdown("""
    <style>
    /* 1. Sidebar Bleue */
    [data-testid="stSidebar"] {
        background-color: #1e3a8a !important;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* 2. Contenu principal Orange */
    .stApp {
        background: linear-gradient(135deg, #fff5e6 0%, #ffe0b3 100%);
    }
    .stApp, .stApp h1, .stApp p, .stApp div, .stApp label {
        color: #000000 !important;
    }
    
    /* 3. Titre */
    h1 {
        text-align: center;
        background: rgba(255, 255, 255, 0.7);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* 4. Bouton Jaune */
    div.stButton > button:first-child {
        background-color: #f59e0b !important;
        color: #000000 !important;
        font-weight: bold !important;
        border: 2px solid #b45309 !important;
        border-radius: 10px !important;
    }
    
    /* 5. Cartes résultats */
    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #f59e0b;
    }
    </style>
    """, unsafe_allow_html=True)

 

model, scaler, df = load_assets()
df_zimbabwe = df[df['Entity'] == 'Zimbabwe'].dropna(subset=['Coal'])

# --- SIDEBAR ---
st.sidebar.header("⚙️ Contrôles")
annee = st.sidebar.slider("Année de prédiction :", 2026, 2050, 2030)

# --- UI PRINCIPALE ---
st.title("⚡ Expert Energy Predictor")

# --- BARRE DE NAVIGATION MODIFIÉE ---
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
    
    # 1. Configuration du style professionnel
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 2. Tracé de l'historique
    sns.lineplot(data=df_zimbabwe, x='Year', y='Coal', 
                 marker='o', color='#1e3a8a', linewidth=2.5, ax=ax, label='Historique')
    
    # 3. Gestion dynamique de la prédiction
    if 'last_pred' in st.session_state:
        # Affichage du point de prédiction
        ax.plot(st.session_state.last_year, st.session_state.last_pred, 
                marker='*', markersize=18, color='#f59e0b', 
                linestyle='None', label='Prédiction', markeredgecolor='white')
        
        # Ligne en pointillés reliant le dernier point réel à la prédiction
        last_real_year = df_zimbabwe['Year'].iloc[-1]
        last_real_val = df_zimbabwe['Coal'].iloc[-1]
        ax.plot([last_real_year, st.session_state.last_year], 
                [last_real_val, st.session_state.last_pred], 
                linestyle='--', color='#f59e0b', alpha=0.6)

    # 4. Ajustement des axes pour aller jusqu'en 2050
    # On fixe la limite à 2051 pour laisser un peu d'espace visuel après 2050
    ax.set_xlim(df_zimbabwe['Year'].min(), 2051)
    
    # 5. Gestion propre des années sur l'axe X
    # On crée une liste d'années cohérente pour l'affichage
    years_ticks = list(range(int(df_zimbabwe['Year'].min()), 2052, 2)) 
    ax.set_xticks(years_ticks)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    
    # 6. Titres et Design
    ax.set_title("Évolution et Projection de la Production (Zimbabwe)", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Année", fontsize=12, labelpad=10)
    ax.set_ylabel("Production (GWh)", fontsize=12, labelpad=10)
    ax.legend(frameon=True, loc='upper left')
    
    sns.despine()
    
    # 7. Affichage
    st.pyplot(fig, clear_figure=True)
    
     

with tab3:
    st.subheader("Objectifs du projet")
    col1, col2, col3 = st.columns(3)
    col1.info("🔍 **Analyse** : Étude des tendances historiques.")
    col2.info("🔮 **Prévision** : Estimation par modèle SVR.")
    col3.info("⚖️ **Décision** : Aide à l'investissement.")

with tab4:
    st.subheader("👤 Identifiants du Développeur")
    st.write("Veuillez saisir vos informations de contact :")
    
    # Formulaire de saisie
    user_nom = st.text_input("Nom complet :")
    user_contact = st.text_input("Numéro de téléphone (Contact) :")
    user_gmail = st.text_input("Adresse Gmail :")
    
    if st.button("Valider mes informations", key="btn_login"):
        if user_nom and user_contact and user_gmail:
            # Vérification simple de présence de '@' dans le mail
            if "@gmail.com" in user_gmail:
                st.success(f"Informations enregistrées pour {user_nom} !")
                st.write(f"📞 Contact : {user_contact}")
                st.write(f"📧 Email : {user_gmail}")
            else:
                st.error("Veuillez saisir une adresse Gmail valide.")
        else:
            st.warning("Veuillez remplir tous les champs.")
            
