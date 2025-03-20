import boto3
import time
from datetime import datetime
import os
import re
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# Initialize DynamoDB client
dynamodb = boto3.client('dynamodb', region_name='us-east-1')

# Store the timestamp of the last fetch
last_fetched_timestamp = int(time.time())  # For example, start with current time

class DataRetriever:
    last_fetched_timestamp = int(time.time())

    @staticmethod
    def fetch_new_data():
        try:
            response = dynamodb.scan(
                TableName='YourTableName',
                FilterExpression='timestamp > :last_fetched',
                ExpressionAttributeValues={
                    ':last_fetched': {'N': str(DataRetriever.last_fetched_timestamp)}
                }
            )
            items = response.get('Items', [])
            print(f"Fetched {len(items)} new records.")
        
            if items:
                # Update the timestamp to the latest one from fetched data
                DataRetriever.last_fetched_timestamp = max(int(item['timestamp']['N']) for item in items)
            return items
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    @staticmethod
    def poll_new_data(interval=60):
        while True:
            new_items = DataRetriever.fetch_new_data()
            # Process new_items as needed here
            time.sleep(interval)

# Directory where raw data is stored
raw_data_path = "C:/Users/jmani/Documents/BatteryMLProject/src/data/four_files/*.csv"
raw_data_files = glob.glob(raw_data_path)
raw_data_path_single_file = "C:/Users/jmani/Documents/BatteryMLProject/src/data/four_files/orionbms_log_2024-03-07-09-47-31.csv"

# Directory where processed data is stored
processed_data_dir = "C:/Users/jmani/Documents/BatteryMLProject/src/data/complete_dataset/processed_data"

class RawDataPlotter:
    def __init__(self, file_path, processed_data_dir):
        self.file_path = file_path
        self.processed_data_dir = processed_data_dir
        self.df = pd.read_csv(self.file_path)
        
        # Convert time column to datetime format
        self.df['Time'] = pd.to_datetime(self.df['Time'], format='%a %b %d %H:%M:%S EST %Y', errors='coerce')

        # Create directory for saving plots
        self.raw_plot_dir = Path("C:/Users/jmani/Documents/BatteryMLProject/src/data/raw_data_plots")
        self.raw_plot_dir.mkdir(parents=True, exist_ok=True)

    def plot_columns(self):
        """Generate and save scatter plots for selected columns."""
        columns_to_plot = ['Pack Voltage', 'Pack Amperage (Current)', 'Pack State of Charge (SOC)']
        file_stem = Path(self.file_path).stem 
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
            save_path = self.raw_plot_dir / f"{file_stem}_{safe_column_name}_over_time.png"
            plt.savefig(save_path)
            print(f"Saved plot: {save_path}")
            plt.close()
        print(f"All plots generated successfully for {file_stem}.")

class PreModelDataProcessor:
    """Class for preprocessing time-series data for XGBoost model."""

    @staticmethod
    def fit_global_scalers(dataframes):
        """Fit global MinMaxScalers on all data files to ensure consistent scaling."""
        all_time_values = []
        all_soc_values = []

        for file, df in dataframes.items():
            df = df.dropna(subset=['Time'])
            df['Time'] = pd.to_datetime(df['Time'], format='%a %b %d %H:%M:%S EST %Y', errors='coerce')
            df = df.dropna(subset=['Time'])  # Drop rows where conversion failed
            elapsed = (df['Time'] - df['Time'].min()).dt.total_seconds().values
            all_time_values.append(elapsed)
            all_soc_values.append(df['Pack State of Charge (SOC)'].values)

        all_time_values = np.concatenate(all_time_values).reshape(-1, 1)
        all_soc_values = np.concatenate(all_soc_values).reshape(-1, 1)

        time_scaler = MinMaxScaler(feature_range=(0, 1)).fit(all_time_values)
        soc_scaler = MinMaxScaler(feature_range=(0, 1)).fit(all_soc_values)

        return time_scaler, soc_scaler

    @staticmethod
    def preprocess_data(df, time_scaler, soc_scaler):
        """Preprocess time-series data using the pre-fitted scalers."""
        df = df.dropna(subset=['Time'])
        # Note: Consider unifying the format or using a flexible parser
        df['Time'] = pd.to_datetime(df['Time'], format='%a %b %d %H:%M:%S EST %Y', errors='coerce')
        df = df.dropna(subset=['Time'])
        
        time_seconds = (df['Time'] - df['Time'].min()).dt.total_seconds().values.reshape(-1, 1)
        soc_values = df['Pack State of Charge (SOC)'].values.reshape(-1, 1)

        scaled_time = time_scaler.transform(time_seconds)
        scaled_soc = soc_scaler.transform(soc_values)

        return np.hstack((scaled_time, scaled_soc))

    @staticmethod
    def prepare_xgboost_data(scaled_data, time_step=60):
        """Prepare time-series data for XGBoost (lags as features)."""
        X, y = [], []
        for i in range(time_step, len(scaled_data)):
            X.append(scaled_data[i - time_step:i, :].flatten())
            y.append(scaled_data[i, 1])  # Use SOC as the target
        return np.array(X), np.array(y)
    
def main():
    
    if not raw_data_files:
        print("No files loaded. Exiting.")
        return

    for file in raw_data_files:
        print(f"Generating raw data plots for {file}...")
        raw_plotter = RawDataPlotter(file, processed_data_dir)
        raw_plotter.plot_columns()

    """
    # Fit global scalers using all available CSV data
    time_scaler, soc_scaler = PreModelDataProcessor.fit_global_scalers(dataframes)
    print("Time Scaler Min, Max:", time_scaler.data_min_, time_scaler.data_max_)
    print("SOC Scaler Min, Max:", soc_scaler.data_min_, soc_scaler.data_max_)

    # Preprocess data for each file and prepare XGBoost input
    for file, df in dataframes.items():
        scaled_data = PreModelDataProcessor.preprocess_data(df, time_scaler, soc_scaler)
        X, y = PreModelDataProcessor.prepare_xgboost_data(scaled_data, time_step=60)
        print(f"File: {file} - Features shape: {X.shape}, Target shape: {y.shape}")

    # Optionally, start polling DynamoDB for new data
    # DataRetriever.poll_new_data()  # Uncomment if you want to start polling"
    """

if __name__ == '__main__':
    main()