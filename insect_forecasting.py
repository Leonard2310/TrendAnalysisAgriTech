import streamlit as st
import requests
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

# Configurazione della pagina Streamlit (deve essere la prima istruzione di Streamlit)
st.set_page_config(page_title="Insect Capture Prediction", page_icon=":insect:", layout="wide")

# Imposta il logger di TensorFlow per mostrare solo errori
tf.get_logger().setLevel('ERROR')

# Percorso del modello LSTM salvato
model_path = 'Inserisci il modello LSTM salvato'

# Funzione per ottenere dati meteorologici storici
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
            'end': timestamp + 86400,  # Fine giornata
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

# Carica il modello LSTM
model = load_lstm_model(model_path)
if model is None:
    st.error(f"Il file del modello non esiste: {model_path}")

# Funzione per la previsione delle catture insetti
def live_forecasting():
    st.title("Live LSTM Insect Capture Forecasting")

    # API Key di OpenWeatherMap
    api_key = st.text_input("Inserisci la tua API Key OpenWeatherMap", type="password")

    # Input della città
    city = st.text_input("Inserisci il nome della città")

    # Input delle date
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Data iniziale")
    with col2:
        end_date = st.date_input("Data finale")

    if st.button("Ottieni previsioni catture insetti"):
        if api_key and city and start_date and end_date:
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
                    'appid': api_key
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
                    weather_data = get_historical_weather_data(lat, lon, start_date, end_date, api_key)
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
                        st.line_chart(lagged_df.set_index('date')['predictions'])
                        progress_bar.progress(100)
                        status_text.text("Operazione completata!")
                    else:
                        st.error("Impossibile ottenere i dati meteo storici.")
                else:
                    st.error("Impossibile determinare latitudine e longitudine per la città.")
        else:
            st.error("Per favore, inserisci tutti i campi richiesti.")

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