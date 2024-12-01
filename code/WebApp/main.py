import streamlit as st
from dotenv import load_dotenv
import os

# Configurazione della pagina Streamlit (deve essere la prima istruzione di Streamlit)
st.set_page_config(page_title="Insect Capture Prediction", page_icon=":beetle:", layout="wide")

from forecast import live_forecasting
from graphs import train_graphs

# Carica le variabili dal file .env
load_dotenv("keyDocker.env")

# Utilizza le variabili
CSV_PATH = os.getenv("CSV_PATH")
MODEL_PATH = os.getenv("MODEL_PATH")

# Verifica che le variabili siano state caricate correttamente
if not CSV_PATH or not MODEL_PATH:
    st.error("Errore: le variabili CSV_PATH o MODEL_PATH non sono state caricate correttamente dal file .env.")
    st.stop()

# Barra laterale per navigare tra le pagine
st.sidebar.title("Navigazione")
page = st.sidebar.radio("Scegli una pagina", ("Live Forecasting", "Grafici dei Modelli"))

# Navigazione tra le pagine
if page == "Live Forecasting":
    live_forecasting(MODEL_PATH)
elif page == "Grafici dei Modelli":
    train_graphs(CSV_PATH)
