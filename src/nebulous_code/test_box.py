import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from charge_test import plot_charge_test

def main():
    # Directory where live test data is stored
    directory = "C:/Users/jmani/Documents/BatteryMLProject/src/data/BMS_Data/live_tests/orionbms_log_2025-02-07-21-19-32testdata1.csv"

    # Read CSV file
    df = pd.read_csv(directory)

    # Convert time column to handle EST timestamps
    df['Time'] = pd.to_datetime(df['Time'], format='%a %b %d %H:%M:%S EST %Y')

    # Create new figure
    plt.figure(figsize=(10, 6))

    # Plot SOH data
    plt.scatter(df['Time'], df["Pack State of Health (SOH)"], alpha=0.5)
    plt.xlabel("Time")
    plt.ylabel("State of Health (SOH)")
    plt.grid(True)
    plt.title("Pack State of Health Over Time")
    plt.savefig(f"{Path('C:/Users/jmani/Documents/BatteryMLProject/src/data').stem}_SOH_over_time.png") #Save figure

    # Plot data
    # plot_charge_test(df)

if __name__ == "__main__":
    main()