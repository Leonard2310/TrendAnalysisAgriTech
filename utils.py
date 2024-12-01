import requests
import os
import streamlit as st
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# Funzione per caricare il modello
@st.cache_resource
def load_lstm_model(model_path):
    if not os.path.exists(model_path):
        return None
    return keras.models.load_model(model_path)

# Funzione per creare feature laggate
def create_lagged_features(df, n_lags, target_col, exog_cols):
    lagged_df = df.copy()
    for col in [target_col] + exog_cols:
        for i in range(1, n_lags + 1):
            lagged_df[f'{col}_lag_{i}'] = lagged_df[col].shift(i)
    lagged_df = lagged_df.dropna()
    return lagged_df

# Funzione per ottenere suggerimenti di città
def get_city_suggestions(query, api_key):
    url = "http://api.openweathermap.org/geo/1.0/direct"
    params = {'q': query, 'limit': 5, 'appid': api_key}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return [f"{item['name']}, {item['country']}" for item in data]
    return []

# Funzione per ottenere dati meteo storici
def get_historical_weather_data(lat, lon, start_date, end_date, api_key):
    weather_data = []
    date = start_date
    while date <= end_date:
        timestamp = int(datetime.combine(date, datetime.min.time()).timestamp())
        url = "http://history.openweathermap.org/data/2.5/history/city"
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
        except:
            return None
        date += timedelta(days=1)
    return weather_data

# Funzione per i simboli meteo
def get_weather_symbol(weather_id):
    if weather_id // 100 == 2:
        return '⛈️'  # Temporale
    elif weather_id // 100 == 3:
        return '🌦️'  # Pioviggine
    elif weather_id // 100 == 5:
        return '🌧️'  # Pioggia
    elif weather_id // 100 == 6:
        return '❄️'  # Neve
    elif weather_id // 100 == 7:
        return '🌫️'  # Nebbia
    elif weather_id == 800:
        return '☀️'  # Sereno
    elif weather_id == 801:
        return '🌤️'  # Poco nuvoloso
    elif weather_id == 802:
        return '⛅'   # Parzialmente nuvoloso
    elif weather_id == 803 or weather_id == 804:
        return '☁️'  # Nuvoloso
    else:
        return '🌡️'  # Altro

# Funzione per visualizzare grafici di training
def training_graphs():
    st.title("Grafici di Addestramento")
    st.write("Visualizza i grafici di addestramento e validazione del modello.")
