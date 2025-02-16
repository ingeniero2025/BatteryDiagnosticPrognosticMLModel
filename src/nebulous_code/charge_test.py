import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Directory where live test data is stored
directory = "C:/Users/jmani/Documents/BatteryMLProject/src/data/BMS_Data/live_tests/orionbms_log_2025-02-07-21-19-32testdata1.csv"

# Read CSV file
df = pd.read_csv(directory)

# Convert time column to handle EST timestamps
df['Time'] = pd.to_datetime(df['Time'], format='%a %b %d %H:%M:%S EST %Y')

def plot_charge_test(df):

    # Create new figure
    plt.figure(figsize=(10, 6))

    # Plot data
    plt.scatter(df['Time'], df["Pack State of Charge (SOC)"], alpha=0.5)
    plt.xlabel("Time")
    plt.ylabel("State of Charge (SOC)")
    plt.grid(True)
    plt.title("Pack State of Charge Over Time")

    # Save figure
    # plt.savefig(f"{Path('C:/Users/jmani/Documents/BatteryMLProject/src/data').stem}.png")

    plt.show()
    
    return plt

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions.squeeze(1)
    
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def preprocess_data(df):
    """ Preprocess the Time and Pack State of Charge (SOC) data from DataFrame for LSTM input """
    # Convert timestamps to numerical values
    time_seconds = (df['Time'] - df['Time'].min()).dt.total_seconds().values.reshape(-1, 1)
    soc_values = df['Pack State of Charge (SOC)'].values.reshape(-1, 1)
    
    # Normalize the data using MinMaxScaler
    time_scaler = MinMaxScaler(feature_range=(0, 1))
    soc_scaler = MinMaxScaler(feature_range=(0, 1))

    # Scale both columns
    scaled_time = time_scaler.fit_transform(time_seconds)
    scaled_soc = soc_scaler.fit_transform(soc_values)

    # Combine scaled data
    scaled_data = np.hstack((scaled_time, scaled_soc))
    
    return scaled_data, (time_scaler, soc_scaler)

def prepare_lstm_data(scaled_data, time_step=60):
    """ Prepare data for LSTM (X, y split) """
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Reshape X to be compatible with LSTM (samples, time steps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y

def train_model(model, train_loader, num_epochs=50, device='cpu'):
    """ Train the PyTorch LSTM model """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training loop
    model.train()
    best_loss = float('inf')
    patience = 10
    counter = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    return model

def plot_charge_test_predictions(df, model, scalers, time_step=60, device='cpu'):

    # Plot data for visualization
    plt = plot_charge_test(df)

    # Prepare data for LSTM prediction (similar to training data preparation)
    scaled_data, _ = preprocess_data(df)
    X, _ = prepare_lstm_data(scaled_data, time_step)
    X_tensor = torch.FloatTensor(X).to(device)

    # Predict the next data points for 'Pack State of Charge (SOC)' using LSTM
    column_name = 'Pack State of Charge (SOC)'
    device = 'cpu'

    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()

    # Reshape predictions to 2D before inverse transform
    predictions_reshaped = predictions.reshape(-1, 1)  # Reshape to (n_samples, 1)

    # Inverse transform the predictions (using the reshaped array)
    predicted_soc = scalers[1].inverse_transform(predictions_reshaped)

    # Plot actual vs. predicted SOC
    fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure and an axes object
    ax.plot(df['Time'][time_step:], df['Pack State of Charge (SOC)'].values[time_step:], label='Actual SOC', color='green')
    ax.plot(df['Time'][time_step:], predicted_soc, label='Predicted SOC', color='red')
    ax.set_xlabel('Time')
    ax.set_ylabel('State of Charge (%)')
    ax.set_title('Actual vs. Predicted SOC')
    ax.legend()
    plt.savefig(f"{Path('C:/Users/jmani/Documents/BatteryMLProject/src/data').stem}_predictions.png")
    plt.show()


if __name__ == "__main__":
    # Handle potential errors during time conversion
    if df['Time'].isnull().any():
        print("Error: Time conversion failed. Check the time format in your CSV file.")
        exit()

    plot_charge_test(df)

    # Preprocess data *before* training the model
    scaled_data, scalers = preprocess_data(df)
    time_step = 60
    X, y = prepare_lstm_data(scaled_data, time_step)
    dataset = TimeSeriesDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = LSTMModel(input_size=1, hidden_size=50, output_size=1)
    model = train_model(model, train_loader, num_epochs=50)

    plot_charge_test_predictions(df, model, scalers, time_step)  # Pass the trained model and scalers



    
    

