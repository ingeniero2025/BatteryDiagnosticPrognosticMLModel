import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# Directory where raw data is stored
raw_data_path = Path("C:/Users/jmani/Documents/BatteryMLProject/src/data/complete_dataset/raw_data/orionbms_log_2024-03-07-09-47-31.csv")

# Directory where processed data is stored
processed_data_dir = Path("C:/Users/jmani/Documents/BatteryMLProject/src/data/complete_dataset/processed_data")

class RawDataPlotter:
    def __init__(self, raw_data_path, processed_data_dir):
        self.raw_data_path = raw_data_path
        self.processed_data_dir = processed_data_dir
        self.df = pd.read_csv(self.raw_data_path)
        # Convert time column to datetime format
        self.df['Time'] = pd.to_datetime(self.df['Time'], format='%a %b %d %H:%M:%S EST %Y', errors='coerce')
        # Create directory for saving plots
        self.raw_plot_dir = Path("C:/Users/jmani/Documents/BatteryMLProject/src/data/raw_data_plots")
        self.raw_plot_dir.mkdir(parents=True, exist_ok=True)

    def plot_columns(self):
        """Generate and save scatter plots for selected columns."""
        columns_to_plot = ['Pack Voltage', 'Pack Amperage (Current)', 'Pack State of Charge (SOC)']
        for column in columns_to_plot:
            if column not in self.df.columns:
                continue
            plt.figure(figsize=(12, 7))
            plt.scatter(self.df['Time'], self.df[column], alpha=0.5)
            plt.xlabel("Time")
            plt.ylabel(column)
            plt.grid(True)
            plt.title(f"{column} Over Time")
            safe_column_name = re.sub(r'[\\/*?:"<>|()\s=]', '_', column)
            save_path = self.raw_plot_dir / f"{safe_column_name}_over_time.png"
            plt.savefig(save_path)
            print(f"Saved plot: {save_path}")
            plt.close()
        print("All plots generated successfully.")

class PreModelDataProcessor:
    """Class for preprocessing data for ML models."""
    @staticmethod
    def preprocess_data(df):
        """Preprocess Time & Pack SOC for LSTM input."""
        df = df.dropna(subset=['Time'])
        df['Time'] = pd.to_datetime(df['Time'], format='%a %b %d %H:%M:%S EST %Y', errors='coerce')
        time_seconds = (df['Time'] - df['Time'].min()).dt.total_seconds().values.reshape(-1, 1)
        soc_values = df['Pack State of Charge (SOC)'].values.reshape(-1, 1)
        time_scaler = MinMaxScaler((0, 1))
        soc_scaler = MinMaxScaler((0, 1))
        scaled_time = time_scaler.fit_transform(time_seconds)
        scaled_soc = soc_scaler.fit_transform(soc_values)
        return np.hstack((scaled_time, scaled_soc)), (time_scaler, soc_scaler)

    @staticmethod
    def prepare_lstm_data(scaled_data, time_step=60):
        """Prepare LSTM data (X, y split)."""
        X, y = [], []
        for i in range(time_step, len(scaled_data)):
            X.append(scaled_data[i - time_step:i, 0])  # Time as input
            y.append(scaled_data[i, 1])  # Predict SOC
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        return X, y

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset class for LSTM Time-Series Data."""
    def __init__(self, X, y, device='cpu'):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.linear(lstm_out[:, -1, :]).squeeze(1)

class ModelTrainer:
    """Train an LSTM Model."""
    @staticmethod
    def train_model(model, train_loader, num_epochs=50, device='cpu'):
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        best_loss = float('inf')
        patience = 10
        counter = 0
        train_losses = []
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            avg_epoch_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_epoch_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        return model, train_losses

class PredictionPlotter:
    """Class for generating prediction plots from a trained LSTM model."""
    def __init__(self, model, scalers, device='cpu'):
        self.model = model.to(device)
        self.scalers = scalers  # Tuple of (time_scaler, soc_scaler)
        self.device = device

    def plot_charge_test_predictions(self, df, time_step=60, save_path=None):
        """Plot actual vs. predicted 'Pack State of Charge (SOC)'."""
        print("Generating predictions...")
        self.model.eval()
        scaled_data, _ = PreModelDataProcessor.preprocess_data(df)
        X, _ = PreModelDataProcessor.prepare_lstm_data(scaled_data, time_step)
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        predicted_soc = self.scalers[1].inverse_transform(predictions.reshape(-1, 1))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['Time'].iloc[time_step:], df['Pack State of Charge (SOC)'].iloc[time_step:], label='Actual SOC', color='green')
        ax.plot(df['Time'].iloc[time_step:], predicted_soc, label='Predicted SOC', color='red')
        ax.set_xlabel('Time')
        ax.set_ylabel('State of Charge (%)')
        ax.set_title('Actual vs. Predicted SOC')
        ax.legend()
        if save_path:
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        plt.show()

class ModelEvaluator:
    """Class for evaluating the model performance."""
    @staticmethod
    def calculate_scaled_mse(model, data_loader, scaler, device='cpu'):
        """Calculate MSE scaled between 0 and 1."""
        model.eval()
        criterion = nn.MSELoss()
        total_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
        mse = total_loss / len(data_loader)
        # Scale MSE based on the range of SOC values
        min_soc = scaler.data_min_[0]
        max_soc = scaler.data_max_[0]
        scaled_mse = mse / ((max_soc - min_soc)**2)
        return scaled_mse

def main():
    # Step 1: Plot raw data using RawDataPlotter
    print("Generating raw data plots...")
    raw_plotter = RawDataPlotter(raw_data_path, processed_data_dir)
    raw_plotter.plot_columns()

    # Step 2: Load dataset & preprocess using PreModelDataProcessor
    print("Preprocessing data...")
    df = pd.read_csv(raw_data_path)
    df['Time'] = pd.to_datetime(df['Time'], format='%a %b %d %H:%M:%S EST %Y', errors='coerce')
    df = df.dropna(subset=['Time'])
    scaled_data, scalers = PreModelDataProcessor.preprocess_data(df)

    # Step 3: Prepare LSTM training data
    print("Preparing LSTM training data...")
    time_step = 60  # Use 60 time steps for prediction
    X, y = PreModelDataProcessor.prepare_lstm_data(scaled_data, time_step)

    # Step 4: Create PyTorch Dataset & DataLoader
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = TimeSeriesDataset(X, y, device=device)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Step 5: Initialize LSTM model
    print("Initializing LSTM model...")
    input_size = 1  # Only using time-series values as input
    hidden_size = 50
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size).to(device)

    # Step 6: Train the model using ModelTrainer
    print("Training model...")
    trained_model, train_losses = ModelTrainer.train_model(model, train_loader, num_epochs=50, device=device)

    # Step 7: Save the trained model
    model_save_path = "C:/Users/jmani/Documents/BatteryMLProject/model/lstm_model.pth"
    Path(os.path.dirname(model_save_path)).mkdir(parents=True, exist_ok=True)
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Step 8: Evaluate the model using ModelEvaluator
    scaled_mse = ModelEvaluator.calculate_scaled_mse(trained_model, train_loader, scalers[1], device=device)
    print(f"Scaled MSE on training data: {scaled_mse:.6f}")

    # Step 9: Generate prediction plots using PredictionPlotter
    print("Plotting predictions...")
    prediction_plotter = PredictionPlotter(trained_model, scalers, device=device)
    prediction_plotter.plot_charge_test_predictions(
        df,
        time_step=time_step,
        save_path="C:/Users/jmani/Documents/BatteryMLProject/src/data/prediction_plots/actual_vs_predicted_soc.png"
    )

if __name__ == "__main__":
    main()
