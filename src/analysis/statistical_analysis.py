import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from pathlib import Path

# Directory where live test data is stored
input_dir = "C:/Users/jmani/Documents/BatteryMLProject/src/data/complete_dataset/raw_data/*.csv"

# Get list of all CSV files in the directory
csv_files = glob.glob(str(input_dir))

# Check if any files were found
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {input_dir}")

# Read and concatenate all CSV files
df_list = [pd.read_csv(file) for file in csv_files]  # Read each file into a DataFrame
df = pd.concat(df_list, ignore_index=True)  # Concatenate all DataFrames

# Reduce memory usage by downcasting data types
for col in df.select_dtypes(include=['float64']).columns:
    df[col] = pd.to_numeric(df[col], downcast="float")

# Convert time column to handle EDT/EST timestamps
try:
    df['Time'] = pd.to_datetime(df['Time'], format='%a %b %d %H:%M:%S %Z %Y', errors='coerce')
except Exception as e:
    print(f"Error converting time column: {e}")
    df['Time'] = pd.to_datetime(df['Time'], format='mixed', errors='coerce')

# Drop invalid time entries
df.dropna(subset=['Time'], inplace=True)

# Output directory
output_dir = Path("C:/Users/jmani/Documents/BatteryMLProject/src/data/BMS_Data/complete_dataset/analysis")
output_dir.mkdir(parents=True, exist_ok=True)

# Save processed data to CSV
df.to_csv(output_dir / "processed_data.csv", index=False)

print(df.head())

# Create a new DataFrame with Time, Voltage, Current and SOC columns
data = pd.DataFrame({
    'Time': df['Time'],
    'Pack Voltage': df['Pack Voltage'],
    'Pack Current': df['Pack Amperage (Current)'],
    'Pack SOC': df['Pack State of Charge (SOC)']
    })

class StatFeatureExtractor:
    """ Class for extracting statistical features from data. """

    def __init__(self, df):
        self.df = df

    def summarize_data(self):
        """ Summarize the data by displaying basic information and descriptive statistics. """
        print(self.df.info())
        print(self.df.describe())

class StatPlotter:

    def __init__(self, title):
        self.title = title

    def create_heat_map(self, data):

        # Compute the correlation matrix
        corr_matrix = data.corr()

        # Plot the correlation heatmap using Seaborn
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap of Voltage, Current, and SOC")

        # Save the figure
        save_path = output_dir / f"{title}_correlation_heat_map.png"
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")

        plt.close()

    def create_histogram(self, data):
    
        # Plot histograms for each column in the DataFrame
        for column in data.columns:

            if column == 'Time':
                continue

            plt.figure(figsize=(10, 6))
            sns.histplot(data[column], kde=True, color='skyblue')
            plt.title(f"{column} Histogram")
            plt.xlabel(column)
            plt.ylabel("Frequency")

            # Save the figure
            safe_column_name = column.replace(" ", "_")
            save_path = output_dir / f"{title}_{safe_column_name}_histogram.png"
            plt.savefig(save_path)
            print(f"Saved plot: {save_path}")

            plt.close() 

    def create_distribution_plot(self, data):

        # Plot distribution plots for each column in the DataFrame
        for column in data.columns:

            if column == 'Time':
                continue

            plt.figure(figsize=(10, 6))
            sns.kdeplot(data[column], color='skyblue', fill=True)
            plt.title(f"{column} Distribution Plot")
            plt.xlabel(column)
            plt.ylabel("Density")

            # Save the figure
            safe_column_name = column.replace(" ", "_")
            save_path = output_dir / f"{title}_{safe_column_name}_distribution_plot.png"
            plt.savefig(save_path)
            print(f"Saved plot: {save_path}")

            plt.close()

    def create_scatterplot(self, data):

        # Create a scatterplot of Pack Voltage vs. Pack SOC
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Pack Voltage', y='Pack SOC', data=data, alpha=0.5)
        plt.title("Pack Voltage vs. Pack SOC")
        plt.xlabel("Pack Voltage")
        plt.ylabel("Pack SOC")

        # Save the figure
        save_path = output_dir / f"{title}_voltage_vs_soc_scatterplot.png"
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")

        plt.close()

        # Create a scatterplot of Pack Current vs. Pack SOC
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Pack Current', y='Pack SOC', data=data, alpha=0.5)
        plt.title("Pack Current vs. Pack SOC")
        plt.xlabel("Pack Current")
        plt.ylabel("Pack SOC")

        # Save the figure
        save_path = output_dir / f"{title}_current_vs_soc_scatterplot.png"
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")

        plt.close()

        # Create a scatterplot of Pack Voltage vs. Pack Current
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Pack Voltage', y='Pack Current', data=data, alpha=0.5)
        plt.title("Pack Voltage vs. Pack Current")
        plt.xlabel("Pack Voltage")
        plt.ylabel("Pack Current")

        # Save the figure
        save_path = output_dir / f"{title}_voltage_vs_current_scatterplot.png"
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")

        plt.close()

    def create_time_series_plot(self, data):

        # Create a time series plot of Pack Voltage
        plt.figure(figsize=(12, 7))
        sns.lineplot(x='Time', y='Pack Voltage', data=data)
        plt.title("Pack Voltage Over Time")
        plt.xlabel("Time")
        plt.ylabel("Pack Voltage")

        # Save the figure
        save_path = output_dir / f"{title}_voltage_over_time.png"
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")

        plt.close()

        # Create a time series plot of Pack Current
        plt.figure(figsize=(12, 7))
        sns.lineplot(x='Time', y='Pack Current', data=data)
        plt.title("Pack Current Over Time")
        plt.xlabel("Time")
        plt.ylabel("Pack Current")

        # Save the figure
        save_path = output_dir / f"{title}_current_over_time.png"
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")

        plt.close()

        # Create a time series plot of Pack SOC
        plt.figure(figsize=(12, 7))
        sns.lineplot(x='Time', y='Pack SOC', data=data)
        plt.title("Pack SOC Over Time")
        plt.xlabel("Time")
        plt.ylabel("Pack SOC")

        # Save the figure
        save_path = output_dir / f"{title}_soc_over_time.png"
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")

        plt.close()

    def create_box_plot(self, data):

        # Create a box plot of Pack Voltage
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Pack Voltage', data=data)
        plt.title("Pack Voltage Box Plot")

        # Save the figure
        save_path = output_dir / f"{title}_voltage_box_plot.png"
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")

        plt.close()

        # Create a box plot of Pack Current
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Pack Current', data=data)
        plt.title("Pack Current Box Plot")

        # Save the figure
        save_path = output_dir / f"{title}_current_box_plot.png"
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")

        plt.close()

        # Create a box plot of Pack SOC
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Pack SOC', data=data)
        plt.title("Pack SOC Box Plot")

        # Save the figure
        save_path = output_dir / f"{title}_soc_box_plot.png"
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")

        plt.close()


if __name__ == "__main__":

    # Instantiate the feature extractor and print summary statistics
    stat_extractor = StatFeatureExtractor(df)
    stat_extractor.summarize_data()

    # Define a title for plots
    title = "Battery Data Analysis"

    """
    # Instantiate and call plotting functions
    stat_plot = StatPlotter(title)
    stat_plot.create_heat_map(data)
    stat_plot.create_histogram(data)
    stat_plot.create_distribution_plot(data)
    stat_plot.create_scatterplot(data)
    stat_plot.create_time_series_plot(data)
    stat_plot.create_box_plot(data)
    
    """


