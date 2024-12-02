import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import os

def train_graphs(folder_path):
    st.title(":chart_with_upwards_trend: Models Graphs")
    st.subheader("Forecasting Results")
    
    # Sottocartelle per i vari csv
    subfolders = sorted([f.path for f in os.scandir(folder_path) if f.is_dir()])
    subfolder_names = [os.path.basename(f) for f in subfolders]
    
    # Creazione di un menu a tendina per selezionare la sottocartella
    selected_folder_name = st.selectbox("Seleziona la cartella", subfolder_names)
    selected_folder = os.path.join(folder_path, selected_folder_name)
    
    csv_files = sorted([f for f in os.listdir(selected_folder) if f.endswith('.csv')])
    
    for csv_file in csv_files:
        if 'results' in csv_file:
            results = pd.read_csv(os.path.join(selected_folder, csv_file), index_col=0, parse_dates=True)

            # Inizializzazionw del titolo del grafico con la prima parola del nome del file
            title = csv_file.split()[0]

            # Ricerca il file metrics corrispondente
            metrics_file = csv_file.replace('results', 'metrics')
            if metrics_file in csv_files:
                metrics = pd.read_csv(os.path.join(selected_folder, metrics_file))
                rmse_train = metrics.loc[metrics['Metric'] == 'RMSE_Train', 'Value'].values
                mae_train = metrics.loc[metrics['Metric'] == 'MAE_Train', 'Value'].values
                rmse_test = metrics.loc[metrics['Metric'] == 'RMSE_Test', 'Value'].values
                mae_test = metrics.loc[metrics['Metric'] == 'MAE_Test', 'Value'].values
                if len(rmse_train) > 0 and len(mae_train) > 0 and len(rmse_test) > 0 and len(mae_test) > 0:
                    title += f"<br>TRAIN:\t\t RMSE: {rmse_train[0]:.2f} \t - \t MAE: {mae_train[0]:.2f}"
                    title += f"<br>TEST:\t\t RMSE: {rmse_test[0]:.2f} \t - \t MAE: {mae_test[0]:.2f}"

            fig = go.Figure()

            # Aggiunta dei valori reali
            fig.add_trace(go.Scatter(x=results.index, y=results['Actual'], mode='lines', name='Actual', line=dict(color='blue')))

            # Aggiunta dei fitted values (train)
            fig.add_trace(go.Scatter(x=results.index, y=results['Fitted_Train'], mode='lines', name='Fitted (Train)', line=dict(color='red', dash='dot')))

            # Aggiunta dei forecast (test)
            fig.add_trace(go.Scatter(x=results.index, y=results['Forecast_Test'], mode='lines', name='Forecast (Test)', line=dict(color='green')))

            # Aggiunta degli intervalli di confidenza
            if all(keyword not in csv_file for keyword in ['DecisionTree', 'GradientBoosting', 'RandomForest']):
                fig.add_trace(go.Scatter(
                    x=results.index.tolist() + results.index[::-1].tolist(),
                    y=results['Upper_CI'].tolist() + results['Lower_CI'][::-1].tolist(),
                    fill='toself',
                    fillcolor='rgba(0,255,0,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval'
                ))

            # Layout
            fig.update_layout(title=title,
                              xaxis_title="Time",
                              yaxis_title="Values",
                              legend=dict(orientation="h"))

            st.plotly_chart(fig)