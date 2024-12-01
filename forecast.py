import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from streamlit_folium import st_folium
import folium
from sklearn.preprocessing import StandardScaler
from utils import load_lstm_model, create_lagged_features, get_city_suggestions, get_historical_weather_data, get_weather_symbol

# Percorso del modello e API Key
model_path = 'lstm_model.keras'
owm_api_key = ''

# Funzione principale per il live forecasting
def live_forecasting():
    st.title(":beetle: Live Insect Capture Forecasting")
    st.subheader("Effettua previsioni di cattura con il modello LSTM")

    # Sezione input
    col_inputs, col_map = st.columns([3, 2])
    with col_inputs:
        api_key = st.text_input("Inserisci la tua API Key OpenWeatherMap", type="password")
        city_input = st.text_input("Inserisci il nome della città")
        if city_input:
            suggestions = get_city_suggestions(city_input, api_key)
            city = st.selectbox("Seleziona la città:", suggestions) if suggestions else city_input

        # Input date
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Data iniziale")
        with col2:
            end_date = st.date_input("Data finale")

        if st.button("Ottieni previsioni"):
            if city and start_date and end_date and api_key:
                if start_date > end_date:
                    st.error("La data iniziale deve essere precedente o uguale alla data finale.")
                else:
                    # Ottieni latitudine e longitudine della città
                    geocode_url = "http://api.openweathermap.org/geo/1.0/direct"
                    params = {'q': city, 'limit': 1, 'appid': api_key}
                    response = requests.get(geocode_url, params=params)
                    data = response.json()
                    if data:
                        lat, lon = data[0]['lat'], data[0]['lon']

                        # Recupera dati meteo storici
                        weather_data = get_historical_weather_data(lat, lon, start_date, end_date, api_key)
                        if weather_data:
                            df = pd.DataFrame(weather_data)
                            df['no. of Adult males'] = 0  # Colonna fittizia per il target
                            lagged_df = create_lagged_features(df, n_lags=5, target_col='no. of Adult males', exog_cols=['temperature', 'humidity'])

                            # Carica modello
                            model = load_lstm_model(model_path)
                            if model:
                                scaler = StandardScaler()
                                lagged_df_scaled = scaler.fit_transform(lagged_df)
                                predictions = model.predict(lagged_df_scaled)
                                st.line_chart(predictions)
                        else:
                            st.error("Impossibile ottenere i dati meteo.")
                    else:
                        st.error("Città non trovata.")

    # Mappa
    with col_map:
        if city and api_key:
            m = folium.Map(location=[41.9028, 12.4964], zoom_start=12)  # Default: Roma
            folium.Marker([41.9028, 12.4964], tooltip=city).add_to(m)
            st_folium(m, width=400, height=300)
