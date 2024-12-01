import streamlit as st
from forecast import live_forecasting
from graphs import train_graphs
from dotenv import load_dotenv
import os

# Carica le variabili dal file .env
load_dotenv("key.env")

# Utilizza le variabili
CSV_PATH = os.getenv("CSV_PATH")

# Barra laterale per navigare tra le pagine
st.sidebar.title("Navigazione")
page = st.sidebar.radio("Scegli una pagina", ("Live Forecasting", "Grafici di Addestramento"))

# Navigazione tra le pagine
if page == "Live Forecasting":
    live_forecasting()
elif page == "Grafici di Addestramento":
    train_graphs(CSV_PATH)
