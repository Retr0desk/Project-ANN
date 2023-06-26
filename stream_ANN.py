import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf 
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

# Load the data
data = pd.read_csv("C:\\Users\\wajit\\Documents\\GitHub\\Project-ANN\\2017-2022copy3.csv", parse_dates=['Tanggal'], date_format='%d/%m/%Y')

checkpoint = ModelCheckpoint('C:\\Users\\wajit\\Documents\\GitHub\\Project-ANN\\checkpoints\\model_1.h5', monitor='val_loss', verbose=1, save_best_only=True)

# Prepare the data
latitude = data["Latitude"]
longitude = data["Longitude"]
kedalaman = data["Kedalaman (km)"]
magnitude = data["Magnitude"]

gempa_semua = np.column_stack((latitude, longitude, kedalaman))
magnitude = magnitude.astype('float64')
gempa_semua = gempa_semua.astype('float64')

X_train, X_test, y_train, y_test = train_test_split(gempa_semua, magnitude, test_size=0.3, random_state=42)
X_valid, X_training, y_valid, y_training = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(3,), kernel_initializer='random_uniform'))
model.add(Dropout(0.2))
model.add(Dense(192, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(48, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')

history = model.fit(X_training, y_training, batch_size=64, epochs=3000, verbose=1, validation_data=(X_valid, y_valid), callbacks=[checkpoint])

st.title("Magnitude Prediction")

# Load the best model
model_path = "C:\\Users\\wajit\\Documents\\GitHub\\Project-ANN\\checkpoints\\model_1.h5"
model = load_model(model_path)
error = model.evaluate(X_valid, y_valid)
error_percentage = error
st.write("Loss: ", error_percentage)

# Create a sidebar
st.sidebar.title("Options User")
koordinat_latitude = st.sidebar.number_input("Enter your latitude coordinate:", value=0.0)
koordinat_longitude = st.sidebar.number_input("Enter your longitude coordinate:", value=0.0)
kedalaman_gempa = st.sidebar.number_input("Enter the earthquake depth (km):", value=0.0)


# Perform prediction
input_gempa = np.array([[koordinat_latitude, koordinat_longitude, kedalaman_gempa]])
prediksi_gempa_user = model.predict(input_gempa)
st.write("Estimated earthquake magnitude: ", prediksi_gempa_user)

# Plot the results
count = 0
predictions = []
for test in X_test:
    koordinat_latitude = float(test[0])
    koordinat_longitude = float(test[1])
    kedalaman_gempa = float(test[2])

    input_data = np.array([[koordinat_latitude, koordinat_longitude, kedalaman_gempa]])
    hasil_prediksi = model.predict(input_data)

    predictions.append(hasil_prediksi.flatten())
    count += 1

    if count == 100:
        break


#Tanggal
start_date = '2023-01-01'
end_date = '2028-12-31'

# Filter the data within the selected time range
filtered_data = data[(data['Tanggal'] >= start_date) & (data['Tanggal'] <= end_date)]





x = data['Tanggal'][:100]
y_data_asli = magnitude[:100]
y_data_forecasted = np.array(predictions).flatten()

plt.figure(figsize=(16, 4))
plt.plot(x, y_data_asli, label='Actual Data')
plt.plot(x, y_data_forecasted, label='Testing Results')
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.title('Actual Data vs Testing Results')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.legend()

# Display the plot in Streamlit
st.pyplot(plt)

# Plot the loss
def model_loss(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    st.pyplot(plt)

model_loss(history)