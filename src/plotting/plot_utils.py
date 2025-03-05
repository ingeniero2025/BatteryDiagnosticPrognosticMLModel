import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
import re

class RawDataPlotter:

    # Directory where test data is stored
    directory = "C:/Users/jmani/Documents/BatteryMLProject/src/data/complete_dataset"

    df = pd.read_csv(directory)

    output_dir = Path("C:/Users/jmani/Documents/BatteryMLProject/src/data/raw_data_plots")

    # Convert time column to handle EST timestamps
    df['Time'] = pd.to_datetime(df['Time'], format='%a %b %d %H:%M:%S EST %Y')

    for column in df.columns:

        if column == 'Time':
            continue

        current_column = df[column]

        # Create new figure
        plt.figure(figsize=(12, 7))

        # Plot SOC data
        plt.scatter(df['Time'], current_column, alpha=0.5)
        plt.xlabel("Time")
        plt.ylabel(column)
        plt.grid(True)
        plt.title(f"{column} Over Time")

        # Sanitize the column name for the filename
        safe_column_name = re.sub(r'[\\/*?:"<>|()\s=]', '_', column)

        # Save the figure
        save_path = output_dir / f"{safe_column_name}_over_time.png"
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")

        plt.close()

    print("All plots generated successfully.")

class PreProcessedDataPlotter:
    # something

    plt.close()

class PredictedDataPlotter:
    # something as well
    
    plt.close()