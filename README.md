# Trend Analysis

This project focuses on predicting the number of pests (insects) captured or their occurrence using a combination of meteorological data and historical pest records. The aim is to develop a robust predictive model that can assist in agricultural management by forecasting pest occurrences based on weather conditions and past data.

## Regression Problem

The primary target variable in this study is the number of captured insects, making it a regression problem. The objective is to predict a continuous quantity—the total number of pests captured within a given timeframe—using various predictive modeling techniques.

## Tools and Techniques

To address this problem, we utilized Python and its extensive suite of libraries for data analysis and modeling, including:
- **pandas** for data manipulation and cleaning
- **matplotlib** and **seaborn** for data visualization
- **scikit-learn** for machine learning algorithms and model evaluation

## Final Objective

The ultimate goal is to create a predictive model that can accurately forecast the number of pests captured based on meteorological data and historical observations. This involves several steps, including data preprocessing, feature engineering, exploratory data analysis, and the application of various statistical and machine learning models.

---

## Data Import

Throughout the project, Excel (HTML) files were converted to CSV format to facilitate automated processing. This conversion made the data more accessible and compatible with data analysis tools. Accurate numerical representation was ensured during this process.

## Data Visualization

### Investigate Data

Initial data visualizations focused on examining the distributions of meteorological variables at the Cicalino and Imola sites. Statistical analyses revealed significant discrepancies between these locations, which were crucial for understanding the data's underlying patterns.

### Target Analysis

The target analysis investigated the frequency of insect captures versus non-capture instances. The data indicated that captures were more frequent at Cicalino, while Imola exhibited fewer capture events.

---

## Preprocessing and Feature Engineering

### Preprocessing

#### Capture CSVs

Data cleaning involved removing irrelevant columns such as "Reviewed" and "Number of Insects," as they were not pertinent to the analysis. Rows containing the value "Cleaning" were also excluded to ensure data quality.

#### Meteorological CSVs

An analytical approach was employed to reduce dimensionality and optimize the data for subsequent analysis. Columns like "low" and "high" temperatures were removed due to their strong correlation with average temperatures, thus simplifying the dataset without losing critical information.

### Clustering

The meteorological dataset had hourly granularity, while insect capture data was recorded daily. To align these datasets, synthetic clustering was used instead of simple daily averages of meteorological variables, ensuring a more accurate representation of the data.

### Dataset Join

After reducing the granularity of the meteorological data through clustering, a join operation was performed based on corresponding dates. This step was necessary to synchronize the datasets for meaningful analysis.

---

## Exploratory Data Analysis (EDA)

The EDA phase followed preprocessing to ensure that the data was consistent and ready for detailed examination.

### Explore Variable Relationships

Scatter plots were utilized to analyze relationships between variables, with point sizes representing insect capture instances. Preliminary correlation studies informed the clustering phase, focusing on the most relevant data for analysis.

### Cross-sectional Data Analysis

Cross-sectional EDA aimed to explore variable relationships through visualizations such as violin plots, which depicted numeric variable distributions and highlighted interconnections between different variables.

### Time Series Analysis

Time series analyses, including autocorrelation (ACF) and partial autocorrelation (PACF), were conducted to evaluate dependencies between current and past observations at specific intervals. This was essential for understanding temporal patterns in the data.

---

## Statistical Learning Techniques

### Background

Based on initial studies and decisions, a modeling approach was adopted to handle the limited data available for each study site individually. The ARIMA (AutoRegressive Integrated Moving Average) method, particularly its ARIMAX variant (ARIMA with exogenous variables), was selected to leverage the predictive power of external factors like meteorological data.

### ARIMAX

#### Overview

The ARIMAX model extends ARIMA by incorporating exogenous variables—external factors that can significantly influence the system. This model was chosen for its ability to handle and predict time series data effectively.

#### Model Customization

For each study site, the ARIMAX model was customized by fine-tuning its parameters:
- **p**: Number of autoregressive terms, reflecting the relationship between an observation and its previous values.
- **d**: Degree of differencing required to make the time series stationary.
- **q**: Number of moving average terms, modeling the relationship between the observation and past errors.

The selection of these parameters was based on analyzing the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) during preliminary phases, followed by further tuning.

#### Training and Testing

The model was trained on 80% of the dataset and tested on the remaining 20%, following standard practices in predictive validation. However, for the Imola 2 site, characterized by an anomalous spike in data, adjustments were made to account for these anomalies.

#### Results

The ARIMAX model demonstrated highly satisfactory results, with predictions achieving relatively low error values measured in terms of Root Mean Square Error (RMSE) and Mean Absolute Error (MAE). One notable limitation was observed in the prediction of null values during the testing phase, which restricted the practical validity of some forecasts. However, an interesting pattern emerged, indicating potential areas for further refinement.

## Machine Learning Techniques

To broaden the study, machine learning approaches were adopted to analyze data from sites with non-stationary historical series. This allowed for the learning of complex patterns present in the data.

### Decision Tree

Decision Trees are predictive models that use a tree structure to iteratively split data based on decision rules derived from features, ultimately reaching leaves that represent the final prediction. This approach is highly interpretable, visualizing the entire decision-making process and identifying which variables influence the result the most.

**Key Features Identified:**
- **Average Humidity 1**: Related to daytime humidity levels, emerged as one of the most influential variables in making predictions.
- **Average Temperature 2**: Associated with nighttime temperatures, showed a significant impact, contributing to splits that notably improved prediction quality.

### Ensemble Learning: Gradient Boosting and Random Forest

Ensemble learning methods combine multiple base models to improve predictive performance. Two approaches were used in this study:
- **Boosting**: Builds base models sequentially, where each new model corrects the errors of the previous one. This approach, exemplified by Gradient Boosting, progressively reduces errors and improves model accuracy.
- **Bagging**: Builds base models independently using random samples from the dataset and then combines the results. An example is Random Forest, which constructs many decision trees with different portions of the data to enhance robustness.

**Experiment Results:**
- **Gradient Boosting**: Achieved excellent results on training data with very low error metrics, thanks to its iterative error reduction. However, test performance was not as satisfactory, indicating potential overfitting.
- **Random Forest**: Showed a trade-off between less perfect training results and improved test performance due to the model's robustness and ability to generalize better.

## Deep Learning Techniques

To further expand the analysis, deep learning models were implemented. These models, characterized by a greater number of parameters than traditional methods, offered the potential to learn complex data representations.

### MLP - Lagged Regressor

The first deep learning approach considered was the MLP (Multilayer Perceptron), a feedforward neural network composed of an input layer, one or more hidden layers, and an output layer. MLP is particularly well-suited for regression tasks.

Training was done using Stochastic Gradient Descent (SGD) with mini-batches, improving computational efficiency compared to traditional gradient descent. The Adam optimizer was also used, combining the best aspects of AdaGrad and RMSProp.

The results with MLP were remarkable, though the risk of overfitting remained. Despite training and test loss not showing clear overfitting, the absence of a validation set made it difficult to fully assess the model's generalization ability.

### LSTM

The LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) designed to handle and predict time sequences, overcoming the limitations of traditional RNNs in managing long-term dependencies.

In this study, all available datasets were used to train the model, inputting not only identified exogenous variables (like temperature and humidity) but also the date as a feature to preserve the temporal aspect of the data.

Training results were promising, and the model learned patterns from the training data. However, there remains a possibility of overfitting. Despite stable test results, further validation is needed to confirm the model's robustness.

## Final Results

In conclusion, the study's results suggest that complex models, such as MLP, tend to perform better on larger datasets where the ability to learn complex representations is more effective. However, in scenarios with limited data, simpler models like ARIMAX and Decision Trees may offer more reliable predictions.

---

## Streamlit Dashboard

### Dashboard Features

A dynamic dashboard was developed using Streamlit, encompassing two main functionalities: real-time prediction of insect captures using a pre-trained predictive model, and visualization of results from various models.

#### Real-time Prediction

The system enables users to select a city, input a date range, and obtain predictions regarding insect captures. Historical meteorological data are retrieved via the OpenWeatherMap API, which provides the necessary input for the predictive model.

Users can view results through an interactive map displaying the selected city’s location and a capture prediction graph, visually representing estimated values over time. The map also shows real-time weather conditions to enhance the context of predictions.

For enhanced security, API keys and necessary application paths are stored in a `.env` configuration file. This approach separates sensitive information from source code, reducing the risk of accidental exposure.

#### Dynamic Graphs

The dashboard includes dynamic visualizations comparing actual and predicted values, complete with confidence intervals and evaluation metrics such as RMSE and MAE. These graphs provide users with a clear understanding of model performance and reliability.

#### Docker Integration

In the subsequent development phase, a Docker container was created to run the Streamlit application, simplifying deployment and execution in an isolated and replicable environment. This ensures that the application can be easily shared and run on different systems without compatibility issues.

##### Prerequisites

Ensure that Docker is correctly installed on your system. You can download and install Docker from the official website.

##### Steps to Use Docker:

1. **Create a Docker Volume for CSV Files**:
   ```
   docker volume create csv-graphs-volume
   ```

2. **Build the Docker Image**:
   Using the Dockerfile in the project directory, build the Docker image, which installs all necessary dependencies and prepares the execution environment:
   ```
   docker build -t streamlit-insect-app .
   ```

3. **Run the Docker Container**:
   Run the Docker container, exposing port 8501 to access the application. Additionally, mount the volume containing the CSV files in the container for direct data management:
   ```
   docker run -p 8501:8501 -v "$(pwd)/csv-graphs:/app/csv-graphs" streamlit-insect-app
   ```

   The application will be accessible at:
   [http://localhost:8501](http://localhost:8501)
