import streamlit as st
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import StandardScaler
from utils import load_lstm_model, create_lagged_features, get_city_suggestions, get_historical_weather_data, get_weather_symbol
import plotly.express as px
from dotenv import load_dotenv
from graphs import train_graphs
import os

# Funzione per ottenere il meteo attuale
def get_current_weather(city_name, owm_api_key):
    url = 'http://api.openweathermap.org/data/2.5/weather'
    params = {
        'q': city_name,
        'appid': owm_api_key,
        'units': 'metric',
        'lang': 'it'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Funzione per la previsione delle catture insetti
def live_forecasting(model_path, owm_api_key):
    st.title(":beetle: OpenBugsWeather")
    st.subheader("Live LSTM Insect Capture Forecasting")
    
    tf.get_logger().setLevel('ERROR')

    # Carica il modello LSTM
    model = load_lstm_model(model_path)
    if model is None:
        st.error(f"Il file del modello non esiste: {model_path}")

    # Organizzazione in due colonne
    col_inputs, col_map = st.columns([3, 2])

    with col_inputs:

        # Input della città con suggerimenti dinamici
        city_input = st.text_input("Inserisci il nome della città", value="Imola")
        if city_input:
            suggestions = get_city_suggestions(city_input, owm_api_key)
            if suggestions:
                city = st.selectbox("Seleziona la città:", suggestions, index=0)
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

        if st.button("Ottieni previsioni catture insetti"):
            if owm_api_key and city and start_date and end_date:
                if start_date > end_date:
                    st.error("La data iniziale deve essere precedente o uguale alla data finale.")
                else:
                    # Barra di progressione
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    status_text.text("Inizio il recupero dei dati meteo...")

                    # Ottieni latitudine e longitudine della città
                    geocode_url = f"http://api.openweathermap.org/geo/1.0/direct"
                    params = {
                        'q': city,
                        'limit': 1,
                        'appid': owm_api_key
                    }
                    try:
                        response = requests.get(geocode_url, params=params)
                        response.raise_for_status()
                        data = response.json()
                        if data:
                            lat = data[0]['lat']
                            lon = data[0]['lon']
                        else:
                            st.error(f"Impossibile trovare la città {city}.")
                            lat, lon = None, None
                    except requests.exceptions.RequestException as e:
                        st.error(f"Errore di connessione: {e}")
                        lat, lon = None, None
                        
                    if lat is not None and lon is not None:
                        # Aggiorna la barra di progresso
                        status_text.text("Recupero dei dati meteo in corso...")
                        progress_bar.progress(50)

                        # Ottieni i dati meteorologici storici
                        weather_data = get_historical_weather_data(lat, lon, start_date, end_date, owm_api_key)
                        if weather_data:
                            progress_bar.progress(75)
                            status_text.text("Previsione in corso...")

                            # Prepara i dati per il modello
                            df = pd.DataFrame(weather_data)
                            df['no. of Adult males'] = 0  # Aggiungi una colonna fittizia per il target

                            # Crea le feature laggate
                            n_lags = 5
                            exog_cols = ['temperature', 'humidity']
                            lagged_df = create_lagged_features(df, n_lags=n_lags, target_col='no. of Adult males', exog_cols=exog_cols)

                            # Prepara i dati di input
                            variables = ['no. of Adult males'] + exog_cols
                            timestep_cols = []
                            for t in range(n_lags):
                                lag = n_lags - t
                                cols = [f'{var}_lag_{lag}' for var in variables]
                                timestep_cols.append(cols)

                            X_list = [lagged_df[cols].values for cols in timestep_cols]
                            input_data = np.stack(X_list, axis=1)

                            # Scala i dati di input
                            scaler = StandardScaler()
                            n_samples, n_timesteps, n_features = input_data.shape
                            input_data_reshaped = input_data.reshape(-1, n_features)
                            input_data_scaled = scaler.fit_transform(input_data_reshaped).reshape(n_samples, n_timesteps, n_features)

                            # Effettua le previsioni
                            predictions = model.predict(input_data_scaled)

                            # Arrotonda le previsioni e convertili in numeri interi
                            predictions = np.clip(predictions, 0, None).round().astype(int)

                            # Aggiungi le previsioni al DataFrame
                            lagged_df['predictions'] = predictions.flatten()

                            # Visualizza il grafico delle previsioni giornaliere
                            fig = px.line(lagged_df, x='date', y='predictions', title='Previsioni')
                            st.plotly_chart(fig)
                            progress_bar.progress(100)
                            status_text.text("Operazione completata!")
                        else:
                            st.error("Impossibile ottenere i dati meteo storici.")
                    else:
                        st.error("Impossibile determinare latitudine e longitudine per la città.")
            else:
                st.error("Per favore, inserisci tutti i campi richiesti.")

    with col_map:
        # Visualizzazione della mappa a destra
        if city and owm_api_key:
            geocode_url = f"http://api.openweathermap.org/geo/1.0/direct"
            params = {'q': city, 'limit': 1, 'appid': owm_api_key}
            try:
                response = requests.get(geocode_url, params=params)
                response.raise_for_status()
                data = response.json()
                if data:
                    lat, lon = data[0]['lat'], data[0]['lon']
                    city_map = folium.Map(location=[lat, lon], zoom_start=10)
                    # Aggiungi il layer meteo di OpenWeatherMap 
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
            
        # Meteo attuale
        weather = get_current_weather(city, owm_api_key)

        # Visualizzazione dei dati meteo con st.metric sotto la mappa
        if weather:
            temperature = weather['main']['temp']
            umidity = weather['main']['humidity']
            weather_id = weather['weather'][0]['id']
            symbol = get_weather_symbol(weather_id)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Temperatura", value=f"{temperature}°C")
            with col2:
                st.metric(label="Umidità", value=f"{umidity}%")
            with col3:
                st.metric(label="Condizioni", value=f"{symbol}")
        else:
            st.write("Impossibile ottenere i dati meteo per la città specificata.")