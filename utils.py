import requests
import os
import streamlit as st
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# Percorso del modello LSTM salvato
model_path = '/Users/l.catello/Library/Mobile Documents/com~apple~CloudDocs/Magistrale Ingegneria Informatica/Information Systems and Business Intelligence/Progetto/Homework 2 - Trend Analysis e Dashboard Streamlit/WebApp/lstm_model.keras'

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
def get_city_suggestions(query, owm_api_key):
    url = "http://api.openweathermap.org/geo/1.0/direct"
    params = {'q': query, 'limit': 5, 'appid': owm_api_key}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        suggestions = [f"{item['name']}, {item['country']}" for item in data]
        return suggestions
    else:
        return []

# Funzionet per ottenere dati meteorologici storici
def get_historical_weather_data(lat, lon, start_date, end_date, owm_api_key):
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
            'appid': owm_api_key
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

# Funzione per mappare le condizioni meteo a simboli
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