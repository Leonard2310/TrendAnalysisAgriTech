import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import os

# Configurazione della pagina Streamlit (deve essere la prima istruzione di Streamlit)
st.set_page_config(page_title="Models Graphs", page_icon=":chart_with_upwards_trend:", layout="wide")

def train_graphs(folder_path):
    st.title(":chart_with_upwards_trend: Models Graphs")
    st.subheader("Forecasting Results - Cicalino 1")
    # Ottieni tutti i file CSV nella cartella
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        # Leggi il CSV
        results = pd.read_csv(os.path.join(folder_path, csv_file), index_col=0, parse_dates=True)

        fig = go.Figure()

        # Aggiungi valori reali
        fig.add_trace(go.Scatter(x=results.index, y=results['Actual'], mode='lines', name='Actual', line=dict(color='blue')))

        # Aggiungi fitted values (train)
        fig.add_trace(go.Scatter(x=results.index, y=results['Fitted_Train'], mode='lines', name='Fitted (Train)', line=dict(color='red', dash='dot')))

        # Aggiungi forecast (test)
        fig.add_trace(go.Scatter(x=results.index, y=results['Forecast_Test'], mode='lines', name='Forecast (Test)', line=dict(color='green')))

        # Aggiungi intervalli di confidenza
        fig.add_trace(go.Scatter(
            x=results.index.tolist() + results.index[::-1].tolist(),
            y=results['Upper_CI'].tolist() + results['Lower_CI'][::-1].tolist(),
            fill='toself',
            fillcolor='rgba(0,255,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        ))

        # Layout
        fig.update_layout(title=f"{csv_file.split('.')[0]} Forecasting Results",
                          xaxis_title="Time",
                          yaxis_title="Values",
                          legend=dict(orientation="h"))

        # Mostra il grafico
        st.plotly_chart(fig)