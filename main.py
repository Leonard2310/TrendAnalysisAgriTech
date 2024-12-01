import streamlit as st
from forecast import live_forecasting
from utils import training_graphs

# Configurazione della pagina Streamlit
st.set_page_config(page_title="Insect Capture Prediction", page_icon=":beetle:", layout="wide")

# Barra laterale per navigare tra le pagine
st.sidebar.title("Navigazione")
page = st.sidebar.radio("Scegli una pagina", ("Live Forecasting", "Grafici di Addestramento"))

# Navigazione tra le pagine
if page == "Live Forecasting":
    live_forecasting()
elif page == "Grafici di Addestramento":
    training_graphs()
