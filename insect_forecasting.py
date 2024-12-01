import streamlit as st
import requests
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import folium
from streamlit_folium import st_folium

# Configurazione della pagina Streamlit (deve essere la prima istruzione di Streamlit)
st.set_page_config(page_title="Insect Capture Prediction", page_icon=":beetle:", layout="wide")

# Imposta il logger di TensorFlow per mostrare solo errori
tf.get_logger().setLevel('ERROR')

# Percorso del modello LSTM salvato
model_path = 'path'

# Funzionet per ottenere dati meteorologici storici
def get_historical_weather_data(lat, lon, start_date, end_date, api_key):
    weather_data = []
    date = start_date
    while date <= end_date:
        timestamp = int(datetime.combine(date, datetime.min.time()).timestamp())
        url = f"http://history.openweathermap.org/data/2.5/history/city"
        params = {
            'lat': lat,
            'lon': lon,
            'type': 'hour',
            'start': timestamp,
            'end': timestamp + 86400,  
            'appid': api_key
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            for item in data['list']:
                weather_data.append({
                    'date': datetime.utcfromtimestamp(item['dt']),
                    'temperature': item['main']['temp'],
                    'humidity': item['main']['humidity']
                })
        except requests.exceptions.RequestException as e:
            st.error(f"Errore di connessione: {e}")
            return None
        except KeyError:
            st.error("Errore nel parsing dei dati.")
            return None
        date += timedelta(days=1)
    return weather_data

# Funzione per caricare il modello LSTM salvato
@st.cache_resource
def load_lstm_model(model_path):
    if not os.path.exists(model_path):
        return None
    model = keras.models.load_model(model_path)
    return model

# Funzione per creare le feature laggate
def create_lagged_features(df, n_lags, target_col, exog_cols):
    lagged_df = df.copy()
    for col in [target_col] + exog_cols:
        for i in range(1, n_lags + 1):
            lagged_df[f'{col}_lag_{i}'] = lagged_df[col].shift(i)
    lagged_df = lagged_df.dropna()
    return lagged_df

# Funzione per suggerire città
def get_city_suggestions(query, api_key):
    url = "http://api.openweathermap.org/geo/1.0/direct"
    params = {'q': query, 'limit': 5, 'appid': api_key}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        suggestions = [f"{item['name']}, {item['country']}" for item in data]
        return suggestions
    else:
        return []

# Carica il modello LSTM
model = load_lstm_model(model_path)
if model is None:
    st.error(f"Il file del modello non esiste: {model_path}")

# Funzione per la previsione delle catture insetti
def live_forecasting():
    st.title(":beetle: OpenBugsWeather")
    st.subheader("Live LSTM Insect Capture Forecasting")

    # Organizzazione in due colonne
    col_inputs, col_map = st.columns([3, 2])

    with col_inputs:
        # API Key di OpenWeatherMap
        api_key = st.text_input("Inserisci la tua API Key OpenWeatherMap", type="password")

        # Input della città con suggerimenti dinamici
        city_input = st.text_input("Inserisci il nome della città")
        if city_input:
            suggestions = get_city_suggestions(city_input, api_key)
            if suggestions:
                city = st.selectbox("Seleziona la città:", suggestions)
            else:
                city = city_input
                st.write("Nessun suggerimento trovato. Utilizzando la città inserita.")
        else:
            city = None

        if city:
            st.write(f"Hai selezionato: {city}")

        # Input delle date
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Data iniziale")
        with col2:
            end_date = st.date_input("Data finale")

        # Bottone per avviare la previsione
        if st.button("Ottieni previsioni catture insetti"):
            st.write("Implementazione del modello di previsione...")
            # Aggiungi qui il resto della logica per il modello

    with col_map:
        # Visualizzazione della mappa a destra
        if city and api_key:
            geocode_url = f"http://api.openweathermap.org/geo/1.0/direct"
            params = {'q': city, 'limit': 1, 'appid': api_key}
            try:
                response = requests.get(geocode_url, params=params)
                response.raise_for_status()
                data = response.json()
                if data:
                    lat, lon = data[0]['lat'], data[0]['lon']
                    city_map = folium.Map(location=[lat, lon], zoom_start=10)
                    # Aggiungi il layer meteo di OpenWeatherMap
                    owm_api_key = 'api key'  
                    weather_tiles = f"https://tile.openweathermap.org/map/precipitation_new/{{z}}/{{x}}/{{y}}.png?appid={owm_api_key}"
                    folium.TileLayer(
                        tiles=weather_tiles,
                        attr='OpenWeatherMap',
                        name='Precipitazioni',
                        overlay=True,
                        control=True
                    ).add_to(city_map)
                    folium.Marker([lat, lon], tooltip=city).add_to(city_map)
                    st_folium(city_map, width=400, height=300)
            except:
                st.write("Mappa non disponibile per questa città.")

# Funzione per la visualizzazione dei grafici di addestramento dei modelli
def training_graphs():
    st.title("Grafici di Addestramento dei Modelli")
    st.write("Qui puoi visualizzare i grafici di addestramento, validazione e analisi dei dati.")

    # Aggiungi i tuoi grafici qui (esempio per il grafico della perdita del modello)
    # st.line_chart(loss_values) 

# Barra laterale per navigare tra le pagine
st.sidebar.title("Navigazione")
page = st.sidebar.radio("Scegli una pagina", ("Live Forecasting", "Grafici di Addestramento"))

# Visualizza la pagina selezionata
if page == "Live Forecasting":
    live_forecasting()
elif page == "Grafici di Addestramento":
    training_graphs()