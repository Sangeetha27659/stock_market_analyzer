import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf  # Replace pandas_datareader with yfinance
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

company = 'AMZN'

# Define the date range
start = dt.datetime(2016, 1, 1)
end = dt.datetime(2024, 1, 1)

# Fetch the stock data using yfinance
data = yf.download(company, start=start, end=end)

# Scaling the 'Close' price data
scaler = MinMaxScaler(feature_range=(0, 1))  # Fix typo here
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days = 60

x_train = []
y_train = []

# Prepare training data
for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Predict the next closing price

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

# Testing the model
test_start = dt.datetime(2024, 1, 1)
test_end = dt.datetime.now()

# Fetch test data using yfinance
test_data = yf.download(company, start=test_start, end=test_end)
actual_prices = test_data['Close'].values

# Prepare inputs for the model
total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)  # Fix reshaping
model_inputs = scaler.transform(model_inputs)

# Prepare test data
x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Make predictions
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot the results
plt.plot(actual_prices, color="black", label=f"Actual {company} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()
# plt.show()

# Predict Next Day
real_data = model_inputs[len(model_inputs) - prediction_days:].reshape(1, prediction_days, 1)

# Make the prediction for the next day
prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction for the next day: {prediction[0][0]}")
plt.show()
