import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
dataset = pd.read_csv("C:\\Users\\wajit\\Documents\\GitHub\\Project-ANN\\2017-2022copy3.csv")

# Convert the 'Tanggal' column to datetime
dataset['Tanggal'] = pd.to_datetime(dataset['Tanggal'])

# Filter the dataset to only include data from '2022-12-1' onwards
dataset = dataset.loc[dataset['Tanggal'] >= '2022-12-1']

# Split the dataset into training and testing sets
train_data = dataset[:-30]
test_data = dataset[-30:]

# Fit the ARIMA model to the training data
model = ARIMA(train_data['Magnitude'], order=(5,1,0))
model_fit = model.fit(disp=0)

# Make predictions for the future
predict_period_dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
forecast_dates = (predict_period_dates - pd.Timestamp('2022-12-1')).days
y_pred_future = model_fit.forecast(steps=30)[0]

# Create a dataframe for the forecast
df_forecast = pd.DataFrame({'Tanggal': predict_period_dates, 'Magnitude': y_pred_future})

# Plot the original data and the forecast
plt.figure(figsize=(12,6))
sns.lineplot(x=train_data['Tanggal'], y=train_data['Magnitude'], label='Original Data')
sns.lineplot(x=df_forecast['Tanggal'], y=df_forecast['Magnitude'], label='Forecast')
plt.title('Forecast of Earthquake Magnitude')
plt.xlabel('Date')
plt.ylabel('Magnitude')
plt.legend()
plt.show()