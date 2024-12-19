# Trend Analysis

The problem addressed in this project involves predicting the number of pests (insects) captured or their occurrence using a combination of meteorological data and historical records of pest occurrences over previous time intervals. This type of analysis falls under predictive analysis applied to temporal and environmental data. Accurately forecasting pest populations is critical for proactive pest management strategies, enabling better resource allocation and reducing the adverse impacts of infestations on agricultural productivity and human health.

## Regression Problem

The target variable in this case is the number of captured insects. It represents a regression problem, where the goal is to predict a continuous quantity—the total number of pests captured within a specific time interval. This task requires a robust modeling approach capable of handling complex relationships between environmental variables and pest activity, ensuring reliable predictions that can inform decision-making processes effectively.

## Tools and Techniques

To tackle this problem, Python and its associated libraries for data analysis and modeling are utilized. These include pandas for data manipulation, matplotlib and seaborn for visualization, and scikit-learn for implementing machine learning models. Additionally, Streamlit is employed to create an interactive dashboard, enabling dynamic and accessible visualization of results and predictions. This user-friendly tool ensures that insights are readily available to stakeholders, enhancing the practical application of the model outcomes.

## Final Objective

The primary goal is to develop a predictive model capable of accurately forecasting the number of pests captured (a regression task) based on meteorological data and prior observations. This involves analyzing historical trends, identifying significant predictors, and creating a reliable forecasting framework. The Streamlit dashboard will provide a user-friendly interface to explore predictions and visualize trends over time. It aims to bridge the gap between complex data-driven insights and practical usability for end-users.

---

## Data Import

Throughout the project, Excel (HTML) files were converted to CSV format to make the data more accessible and compatible with automated processing tools. Numerical values in the original data were represented as strings, with commas used as decimal separators. These values were converted from strings to floats, adopting the dot as the decimal delimiter. This preprocessing step ensured that the data could be efficiently utilized for subsequent analyses without introducing errors or inconsistencies.

## Data Visualization

### Investigate Data

Initial data visualization involved examining the distributions of meteorological variables associated with the Cicalino and Imola sites. Statistical analyses revealed significant discrepancies between the overall distributions of these primary sites, suggesting substantial differences in climatic patterns. However, within subgroups, the distributions of Cicalino 1 and Cicalino 2, as well as Imola 1, Imola 2, and Imola 3, exhibited considerable homogeneity, indicating internal consistency among respective sites. This step was crucial for identifying potential biases and guiding data integration strategies.

### Target Analysis

The target analysis focused on the frequency of insect captures relative to non-capture instances. Data indicated that captures were more frequent for Cicalino, while Imola showed fewer capture observations. Nevertheless, non-capture instances predominated in both cases. This result is statistically justified, as insects are typically captured in limited numbers. Understanding these patterns was essential for defining model objectives and evaluating prediction feasibility.

---

## Preprocessing and Feature Engineering

### Preprocessing

#### Capture CSVs

Data cleaning involved removing irrelevant information. Columns such as "Reviewed" and "Number of Insects" were dropped as they were deemed irrelevant. Rows containing the value "Cleaning" were removed as they represented extraneous data. Additionally, the "Event" column was discarded after removing the rows associated with the "Cleaning" value, resulting in a cleaner dataset. These steps ensured that the input data were free from noise and redundancy, improving model reliability.

#### Meteorological CSVs

An analytical approach was adopted to reduce dimensionality and optimize the data for subsequent analysis. Columns "low" and "high" were removed due to their strong correlation with temperatures. Correlation analyses and PCA conducted with JMP validated this choice, ensuring no significant information loss. Days lacking complete 24-hour observations were eliminated to prevent introducing inconsistencies into clustering. Although data from the same sites showed high correlation, it was decided not to merge them, as insect captures appeared independent under equivalent meteorological conditions. This observation emerged from PCA analyses, which suggested hidden patterns worth exploring via statistical forecasting and machine learning.

### Clustering

The meteorological dataset exhibited hourly granularity, while insect capture data had daily resolution. To align the datasets, instead of simple daily averages of meteorological variables, synthetic features based on hourly intervals were created. These intervals were defined using a non-overlapping sliding window approach. Hierarchical clustering on temperature and humidity variables revealed two main clusters corresponding to primary time intervals. Subsequently, the k-means method refined hourly interval segmentation within identified clusters. The mean temperature and humidity for each cluster were calculated and added to the dataset as new synthetic daily-resolution features. These features captured nuanced meteorological patterns, enhancing the predictive power of the model.

### Dataset Join

After reducing meteorological data granularity through clustering, a join operation was performed based on corresponding dates. The temporal range of the Imola 3 dataset was deemed too short for meaningful trend analysis and was thus excluded. This decision was made to maintain the integrity of the analysis and ensure consistency in the dataset used for modeling.

---

## Exploratory Data Analysis (EDA)

The EDA phase was positioned post-preprocessing, as consistent data were necessary for adequate representation.

### Explore Variable Relationships

Scatter plots were used to analyze relationships between variables, with point size representing insect capture instances. Preliminary correlation studies informed the clustering phase, so only data for Cicalino 1 were displayed. The high number of captures allowed for significant observations of temperature and humidity variations' effects on the target variable. This analysis provided valuable insights into the primary drivers of insect activity.

### Cross-sectional Data Analysis

Cross-sectional EDA aimed to explore variable relationships through visualizations that illustrated data distribution and interconnections. Violin plots showed numeric variable distributions, highlighting density, median, and variability, while correlation matrices emphasized relationships using Pearson, Spearman, and Kendall metrics. Notable findings included moderate positive correlations between "Average Temperature 1" and insect captures and strong correlations between humidity variables. Anomalous high negative correlations observed for Imola 1 suggested potential microclimate conditions or exceptional meteorological events. These observations informed feature selection and model development.

### Time Series Analysis

Time series autocorrelation (ACF) and partial autocorrelation (PACF) analyses were conducted to evaluate dependencies between current and past observations at specific intervals. This was essential for identifying significant lags and isolating direct correlations between observations, paving the way for statistical and machine learning-based forecasting. These analyses provided a deeper understanding of temporal patterns, which is critical for developing accurate prediction models.

---

## Statistical Learning Techniques

### Background

Based on the studies and decisions made during the project, it was deemed appropriate to adopt a modeling approach tailored to handle the limited data available for each study site, treated individually. In this context, statistical models were identified as an optimal solution due to their effectiveness in analyzing small datasets.

Among the available models, the ARIMA (AutoRegressive Integrated Moving Average) method, with a specific focus on its ARIMAX variant (ARIMA with exogenous variables), was chosen to leverage the predictive power gained from integrating environmental variables.

### ARIMAX

#### Overview

The ARIMAX (AutoRegressive Integrated Moving Average with eXogenous variables) model extends ARIMA by incorporating exogenous variables—external factors that can significantly influence the system’s behavior. These additional explanatory variables enhance the model’s ability to predict the target variable. In this project, the exogenous variables used were average temperature and average humidity, derived from meteorological clusters, while the target variable represented new insect captures per event.

#### Model Customization

For each study site, the ARIMAX model was tailored by fine-tuning its parameters:

- **p**: Number of autoregressive terms (AR), reflecting the relationship between an observation and its previous values.
- **d**: Degree of differencing required to make the time series stationary.
- **q**: Number of moving average terms (MA), modeling the relationship between the observation and past errors.

The selection of these parameters was based on analyzing the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) during preliminary phases, followed by a tuning process to refine the model’s accuracy.

#### Training and Testing

The model was trained on 80% of the dataset and tested on the remaining 20%, following standard practices in predictive validation. However, for the Imola 2 site, characterized by an anomalous spike at the end of the series, an 85% training and 15% testing split was adopted. This adjustment ensured that the model was not disadvantaged by the absence of significant fluctuations during the learning phase.

#### Results

The ARIMAX model demonstrated highly satisfactory results. Its predictions achieved relatively low error values, measured in terms of Root Mean Square Error (RMSE) and Mean Absolute Error (MAE). Additionally, the predicted values fell within calculated confidence intervals, confirming the model's reliability.

One notable limitation was observed in the prediction of null values during the testing phase, which restricted the practical validity of some forecasts. However, an interesting aspect emerged: the model correctly learned the stability of exogenous variables during training and translated this into consistent predictions, accurately reflecting the underlying pattern. This capability underscores the model’s effectiveness in capturing stable trends and providing coherent forecasts.

## Machine Learning Techniques
To broaden the study, a machine learning approach was adopted to analyze data from sites whose historical series were non-stationary. This allows us to learn the complexity of the patterns present in the data without requiring transformations for stationarity. By utilizing machine learning models, we can capture non-linear relationships and interactions between variables. Another advantage of this methodology is the ability to analyze the importance of features in a multivariate context. Through model interpretation techniques, we sought to identify which environmental variables (e.g., temperature, humidity) had a significant impact on forecasting insect captures.

### Decision Tree
Decision Trees are predictive models that use a tree structure to iteratively split data based on decision rules derived from features, ultimately reaching leaves that represent the final prediction. Decision Tree Regressors are specialized models used for regression problems where the target value is continuous. The leaves of the tree contain average (or weighted sum) values of the data that fall into them, providing an estimate of the target value.

This approach is highly interpretable because it visualizes the entire decision-making process and identifies which variables influence the result the most. To improve the model's predictive effectiveness, lagged features were created, representing past values of the original variables shifted by one or more intervals. This transformation is essential for time-series problems as it enables the model to account for autocorrelation and temporal patterns in the data.

For each dataset, the resulting decision tree was printed to analyze its structure and gain a better understanding of the rules learned by the model. This visualization offers an intuitive representation of interactions between variables and decisions made to make predictions, providing valuable insights for validating and interpreting the results.

**Key Features Identified:**
- **Average Humidity 1**: Related to daytime humidity levels, emerged as one of the most influential variables in making predictions.
- **Average Temperature 2**: Associated with nighttime temperatures, showed a significant impact, contributing to splits that notably improved the prediction quality.

### Ensemble Learning: Gradient Boosting and Random Forest
Ensemble learning methods combine multiple base models to improve predictive performance. Two approaches were used in this study:
- **Boosting** builds base models sequentially, where each new model corrects the errors of the previous one. This approach, like in the case of Gradient Boosting, progressively reduces errors and improves predictions but can be susceptible to overfitting if not properly regulated.
- **Bagging** builds base models independently using random samples from the dataset and then combines the results. An example is Random Forest, which constructs many decision trees with different portions of data and features, improving the model's generalization and reducing overfitting.

**Experiment Results:**
- **Gradient Boosting**: Excellent results on training data with very low error metrics, thanks to the boosting's ability to reduce errors iteratively. However, test performance was not as satisfactory, indicating some difficulty in generalizing.
- **Random Forest**: A trade-off was observed, with less perfect training results due to bagging sacrificing an optimal fit for robustness. However, test performance improved due to the model's ability to avoid overfitting and generalize better to unseen data.

## Deep Learning Techniques
To further expand the analysis and compare the previous models, deep learning models were also implemented. These models, characterized by a greater number of parameters than traditional methods, offer the ability to learn more complex representations of the data and capture more sophisticated patterns.

Despite the complexities of these models and the risk of overfitting due to the limited amount of available data, an interesting approach was identified to explore and try to extract a common pattern across all datasets, benefiting from deep learning's ability to generalize better even with relatively small datasets.

### MLP - Lagged Regressor
The first deep learning approach considered was the MLP (Multilayer Perceptron), a feedforward neural network composed of an input layer, one or more hidden layers, and an output layer. MLP is particularly suitable for regression and classification problems as it can learn non-linear relationships between variables. In this project, only two hidden layers were used, considering the simplicity of the problem, with the number of neurons adjusted based on the specific dataset.

Training was done using Stochastic Gradient Descent (SGD) with mini-batches, improving computational efficiency compared to traditional gradient descent. The Adam optimizer was also used, combining the advantages of other optimization algorithms (Momentum and RMSprop) to dynamically adjust learning rates and speed up convergence. The Mean Square Error was used as the loss function.

The results with MLP were remarkable, though the risk of overfitting remained. Despite training and test loss not showing clear overfitting, the absence of a validation set made it difficult to fully assess the model’s generalization ability. In practice, it’s hard to be certain that the MLP model generalizes well.

### LSTM
The LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) designed to handle and predict time sequences, overcoming the limitations of traditional RNNs in managing long-term dependencies. The LSTM structure includes memory cells and input, output, and forget gates, allowing it to retain and update information over long periods, making it particularly well-suited for sequential and time-series data.

In this study, all available datasets were used to train the model, inputting not only identified exogenous variables (like temperature and humidity) but also the date as a feature to preserve the temporal component of the data. The Cicalino and Imola 2 datasets were used for training, while Imola 1 was chosen as the test dataset.

Training results were interesting, and the model did learn a pattern from the training data. However, there is still a possibility that the model may have overfitted. Despite stable test results, the train-test loss graph suggests that the model learned a basic pattern applied to the test data. Given the available data, the model seems to have reached its maximum predictive potential, indicating that further improvements would require additional data or model optimization.

## Final Results
In conclusion, the study's results suggest that complex models, such as MLP, tend to perform better on larger datasets where the ability to learn complex representations is more effective. However, in the case of smaller datasets, simpler statistical and machine learning methods showed slightly better accuracy, likely due to their lower propensity for overfitting and greater robustness with limited data. These results suggest that for small datasets, using less complex models that generalize better may be the best approach.

---

This README explains the use of machine learning techniques and models, including decision trees, ensemble methods, and deep learning, to analyze and predict insect capture data.


---

## Streamlit Dashboard

### Dashboard Features

A dynamic dashboard was developed using Streamlit, encompassing two main functionalities: real-time prediction of insect captures using a pre-trained predictive model, and visualization of results from previously trained models through graphical representations.

#### Real-time Prediction

The system enables users to select a city, input a date range, and obtain predictions regarding insect captures. Historical meteorological data are retrieved via the OpenWeatherMap API, which provides detailed climate information such as temperature and humidity to feed into the predictive model. This model is based on a Long Short-Term Memory (LSTM) neural network. The LSTM model allows real-time predictions of insect captures by utilizing historical meteorological data dynamically selected by the user. A dataset from previous projects with extensive data volume was employed to train the model, ensuring robust and consistent performance for accurate and reliable predictions.

Users can view results through an interactive map displaying the selected city’s location and a capture prediction graph, visually representing estimated values over time. The map also shows real-time meteorological data, offering an additional layer of information for users. This interactive feature ensures that stakeholders can make informed decisions based on intuitive and accessible data visualizations.

For enhanced security, API keys and necessary application paths are stored in a `.env` configuration file. This approach separates sensitive information from source code, reducing the risk of accidental exposure in the GitHub repository.

#### Dynamic Graphs

Dynamic visualization of results from various previously used models was implemented. Graphs comparing actual and predicted values include confidence intervals and evaluation metrics such as RMSE and MAE. A dropdown menu allows users to select and view graphs for each dataframe, which are generated from tabular data extracted during model training. These graphs provide a comprehensive overview of model performance, helping users evaluate the effectiveness of different approaches.

#### Docker Integration

In the subsequent development phase, a Docker container was created to run the Streamlit application, simplifying deployment and execution in an isolated and replicable environment. This ensures that the application can be easily shared and deployed across various systems without compatibility issues.

##### Prerequisites

Ensure Docker is correctly installed on your system. You can download and install Docker from the official website.

##### Steps to Use Docker:

1. **Create a Docker Volume for CSV Files**  
   The following command creates a Docker volume to store CSV files generated during application execution:
   ```
   docker volume create csv-graphs-volume
   ```

2. **Build the Docker Image**  
   Using the Dockerfile in the project directory, the following command builds the Docker image, installing all necessary dependencies and preparing the execution environment:
   ```
   docker build -t streamlit-insect-app .
   ```

3. **Run the Docker Container**  
   The following command runs the Docker container, exposing port 8501 to access the application. Additionally, the volume containing the CSV files is mounted in the container for direct data management within the environment:
   ```
   docker run -p 8501:8501 -v "$(pwd)/csv-graphs:/app/csv-graphs" streamlit-insect-app
   ```
   The application will be accessible at:
   [http://localhost:8501](http://localhost:8501)


