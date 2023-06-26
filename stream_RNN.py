import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import tensorflow as tf 
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

# Load the data
data = pd.read_csv("C:\\Users\\wajit\\Documents\\GitHub\\Project-ANN\\2017-2022copy3.csv")

#Checkpoint
checkpoint = ModelCheckpoint('C:\\Users\\wajit\\Documents\\GitHub\\Project-ANN\\checkpoints\\model_5.h5', monitor='val_loss', verbose=1, save_best_only=True)

# Set up the page layout
st.set_page_config(layout="wide")

# Create a sidebar
st.sidebar.title("Options")
n_past = st.sidebar.slider("Number of past days", 10, 30, 30)
n_future = st.sidebar.slider("Number of future days to predict", 1, 365, 365)

# Prepare the data
train_dates = pd.to_datetime(data['Tanggal'])
cols = list(data)[2:6]  # Adjust the range to include the 6th column
data_for_training = data[cols].astype(float)
scaler = StandardScaler()
scaler = scaler.fit(data_for_training)
data_for_training_scaled = scaler.transform(data_for_training)

train_X = []
train_y = []

for i in range(n_past, len(data_for_training_scaled) - n_future + 1):
    train_X.append(data_for_training_scaled[i - n_past:i, :])  # Use all columns for training data
    train_y.append(data_for_training_scaled[i + n_future - 1:i + n_future, -1])  # Use the last column for prediction

train_X, train_y = np.array(train_X), np.array(train_y)

# Build the model
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=True))
model.add(LSTM(units=32, return_sequences=True))
model.add(LSTM(units=16, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.summary()

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')

# Train the model
history = model.fit(train_X, train_y, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[checkpoint])

model = load_model("C:\\Users\\wajit\\Documents\\GitHub\\Project-ANN\\checkpoints\\model_5.h5")
error = model.evaluate(train_X, train_y)
error_percentage = error


# Perform prediction
n_days_for_prediction = n_past + n_future
predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_days_for_prediction, freq='1d').tolist()
prediction = model.predict(train_X[-n_days_for_prediction:])
prediction_copies = np.repeat(prediction, train_X.shape[2], axis=-1)
y_pred_future = scaler.inverse_transform(prediction_copies)[:, -1]
forecast_dates = [time_i.date() for time_i in predict_period_dates]

df_forecast = pd.DataFrame({'Tanggal': np.array(forecast_dates), 'Magnitude': y_pred_future})
df_forecast['Tanggal'] = pd.to_datetime(df_forecast['Tanggal'])

# Prepare the original data
original = data[['Tanggal', 'Magnitude']]
original['Tanggal'] = pd.to_datetime(original['Tanggal'])
original = original.loc[original['Tanggal'] >= '2022-12-1']

# Plot the data using Streamlit
st.title("Magnitude Prediction")
st.write("Loss:", error_percentage)
st.subheader("Original Data")
st.line_chart(original.set_index('Tanggal'))
st.subheader("Forecast Data")
st.line_chart(df_forecast.set_index('Tanggal'))
