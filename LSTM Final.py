import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K

# STEP 1: Load your filtered time series (date + sales)
try:
    df = pd.read_csv("sales_train_subset_part_1.csv)")
except FileNotFoundError:
    print("Error: The dataset file is missing. Please check the file path.")
if 'item_id' not in df.columns or 'store_id' not in df.columns:
    raise ValueError("Required columns are missing from the dataset.")
calendar = pd.read_csv("calendar_subset_part_1.csv)")

# Preprocessing: convert to long format
id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
df_long = pd.melt(df, id_vars=id_vars, var_name='d', value_name='sales')
df_long = df_long.merge(calendar[['d', 'date']], on='d', how='left')
df_long['date'] = pd.to_datetime(df_long['date'])

# Loop through all unique item/store combinations
for item in df_long['item_id'].unique():
    for store in df_long['store_id'].unique():
# Filter data for the current item/store combination
        subset = df_long[(df_long['item_id'] == item) & (df_long['store_id'] == store)]
        subset = subset[['date', 'sales']].sort_values('date')

# STEP 2: Normalize the sales values
scaler = MinMaxScaler()
subset['sales_scaled'] = scaler.fit_transform(subset[['sales']])

# STEP 3: Create sequences for LSTM
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

window_size = 30
sales_values = subset['sales_scaled'].values
X, y = create_sequences(sales_values, window_size)

# Reshape for LSTM input: [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# STEP 4: Build the LSTM model
model = Sequential([
    LSTM(64, activation='tanh', input_shape=(window_size, 1), return_sequences=True), # First LSTM layer
    Dropout(0.2), # Dropout layer
    LSTM(32, activation='tanh'), # Second LSTM layer
    Dense(1) # Dense output layer
])

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

model.compile(optimizer='adam', loss='mse', metrics=['mae', rmse])
model.summary()

# Define learning rate schedule
def lr_schedule(epoch, lr):
    if epoch % 10 == 0 and epoch > 0:
        lr = lr * 0.1 # Decrease learning rate by a factor of 0.1 every 10 epochs
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

# STEP 5: Train the model
history = model.fit(X, y, epochs=50, batch_size=32, verbose=1, validation_split=0.2, callbacks=[lr_scheduler])

# STEP 6: Predict and plot results
y_pred = model.predict(X)
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_actual_rescaled = scaler.inverse_transform(y.reshape(-1, 1))

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(y_actual_rescaled, label='Actual')
plt.plot(y_pred_rescaled, label='Predicted')
plt.title('LSTM Sales Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

# Plotting training loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()