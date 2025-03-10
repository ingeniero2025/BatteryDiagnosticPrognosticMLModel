import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
import re

""" set up automation in the future to periodically receive batches of data from the database """
# Directory where raw data is stored
raw_data_path = Path("C:/Users/jmani/Documents/BatteryMLProject/src/data/complete_dataset/raw_data/orionbms_log_2024-03-07-09-47-31.csv")

# Directory where processed data is stored
processed_data_dir = Path("C:/Users/jmani/Documents/BatteryMLProject/src/data/complete_dataset/processed_data")

class RawDataPlotter:
    def __init__(self, raw_data_path, processed_data_dir):
        self.raw_data_path = raw_data_path
        self.processed_data_dir = processed_data_dir
        self.df = pd.read_csv(self.raw_data_path)

        # Convert time column to handle EST timestamps
        self.df['Time'] = pd.to_datetime(self.df['Time'], format='%a %b %d %H:%M:%S EST %Y')

        # Directory for saving plots
        self.raw_plot_dir = Path("C:/Users/jmani/Documents/BatteryMLProject/src/data/raw_data_plots")
        self.raw_plot_dir.mkdir(parents=True, exist_ok=True)

    def plot_columns(self):
        """Generate and save scatter plots for selected columns."""
        columns_to_plot = [
            'Pack Voltage',
            'Pack Amperage (Current)',
            'Pack State of Charge (SOC)'
        ]

        for column in self.df.columns:
            if column not in columns_to_plot:
                continue

            plt.figure(figsize=(12, 7))
            plt.scatter(self.df['Time'], self.df[column], alpha=0.5)
            plt.xlabel("Time")
            plt.ylabel(column)
            plt.grid(True)
            plt.title(f"{column} Over Time")

            # Sanitize the column name for the filename
            safe_column_name = re.sub(r'[\\/*?:"<>|()\s=]', '_', column)

            # Save the figure
            save_path = self.raw_plot_dir / f"{safe_column_name}_over_time.png"
            plt.savefig(save_path)
            print(f"Saved plot: {save_path}")

            plt.close()

        print("All plots generated successfully.")



    def preprocess(raw_data_dir, processed_data_dir):
        """Processes all CSV files in the input directory and saves them to the output directory."""
        
        file_list = glob.glob(os.path.join(raw_data_dir, "*.csv"))

        if not file_list:
            raise FileNotFoundError(f"No CSV files found in {raw_data_dir}")

        # Create output directory if it doesn't exist
        os.makedirs(processed_data_dir, exist_ok=True)

        processed_files = []

        for file in file_list:
            # Load raw data
            df = DataPreprocessor.load_data(file)
            # Clean data
            df = DataPreprocessor.clean_data(df)

            # Generate output file path
            output_file = os.path.join(processed_data_dir, os.path.basename(file))

            # Save processed data
            df.to_csv(output_file, index=False)
            processed_files.append(output_file)

            print(f"Processed file saved: {output_file}")


        # Read and concatenate all CSV files
        df_list = [pd.read_csv(file) for file in csv_files]  # Read each file into a DataFrame
        df_combined = pd.concat(df_list, ignore_index=True)

        # Reduce memory usage by downcasting data types
        for col in df_combined.select_dtypes(include=['float64']).columns:
            df_combined[col] = pd.to_numeric(df_combined[col], downcast="float")


        # Convert time column to datetime format
        try:
            df_combined['Time'] = pd.to_datetime(df_combined['Time'], format='%a %b %d %H:%M:%S %Z %Y', errors='coerce')
        except Exception as e:
            print(f"Error converting time column: {e}")

        # Drop invalid time entries
        df.dropna(subset=['Time'], inplace=True)

        # Save final processed dataset
        final_output_dir = Path(processed_data_dir) / "final_dataset"
        final_output_dir.mkdir(parents=True, exist_ok=True)
        final_output_path = final_output_dir / "processed_data.csv"

        df_combined.to_csv(final_output_path, index=False)

        print(f"Final combined dataset saved: {final_output_path}")

        print(df.head())

class DataPreprocessor:
    """ Class for preprocessing battery data. """

    def load_data(self, file_path):
        """Load raw data from a CSV file."""
        try:
            df = pd.read_csv(file_path)
            print("Data loaded successfully.")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def clean_data(self, df):
        """Cleans the battery dataset by handling time, missing values, and outliers."""

        if 'Time' not in df.columns:
            print("Error: 'Time' column not found in dataframe.")
            return df  # Return unmodified dataframe

        # Convert time column to datetime
        try:
            df['Time'] = pd.to_datetime(df['Time'], format='%a %b %d %H:%M:%S EST %Y')
        except Exception as e:
            print(f"Error converting time column: {e}")
        
        # Drop rows with invalid time entries
        df.dropna(subset=['Time'], inplace=True)

        # Handle missing values (forward fill then backward fill)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

        # Outlier removal using IQR method
        def remove_outliers_iqr(df, cols):
            for col in cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Apply filter correctly
                df = df.loc[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            return df

        # Apply outlier removal for numeric columns (excluding 'Time')
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            df = remove_outliers_iqr(df, numeric_cols)

        print("Data cleaning complete.")
        return df

# stopped right here 3-6-25 2:37 PM

class PreProcessedDataPlotter:
    # something

    plt.close()

class StatMeasures:
    def __init__(self, processed_data_dir):
        # Load preprocessed data from CSV
        self.df = pd.read_csv(processed_data_dir)
        # Define the specific columns to analyze
        self.columns_of_interest = ['Pack Voltage', 'Pack Amperage (Current)', 'Pack State of Charge (SOC)']

    def calculate_statistics(self):
        stats = {}

        for column in self.columns_of_interest:
            if column in self.df.columns:
                data = self.df[column]

                # Compute statistics
                stats[column] = {
                    "Mean": data.mean(),
                    "Variance": data.var(),
                    "Interquartile Range": data.quantile(0.75) - data.quantile(0.25),
                    "Min": data.min(),
                    "Max": data.max(),
                    "Standard Deviation": data.std()
                }

        # Convert statistics to DataFrame
        stats_df = pd.DataFrame.from_dict(stats, orient='index')
        stats_df.reset_index(inplace=True)
        stats_df.rename(columns={'index': 'Metric'}, inplace=True)

        # Save statistics to CSV
        stats_df.to_csv("statistics_summary.csv", index=False)

        # Compute and save correlation matrix (only for the specified columns)
        correlation_matrix = self.df[self.columns_of_interest].corr()
        correlation_matrix.to_csv("correlation_matrix.csv")

        print("Statistics summary and correlation matrix saved successfully.")

# stopped here 3-6-25 3:27 PM, please validate above sections of code

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

class StatFeatureExtractor:
    """ Class for extracting statistical features from data. """

    def __init__(self, df):
        self.df = df

    def summarize_data(self):
        """ Summarize the data by displaying basic information and descriptive statistics. """
        print(self.df.info())
        print(self.df.describe())

class PreModelDataProcessor:
    """ Class for preprocessing data for machine learning models. """

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

class ModelTrainer:

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

        train_losses = []
    
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

            avg_epoch_loss = epoch_loss / len(train_loader) #Average loss per epoch
            train_losses.append(avg_epoch_loss) #Append average loss to list
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_epoch_loss:.4f}") #Print training loss for each epoch
        
            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
        return model, train_losses
    
class PredictedDataPlotter:
    # something as well
    
    plt.close()

class MSECalculator:

    def calculate_scaled_mse(model, data_loader, scaler, device='cpu'):
        """Calculate MSE scaled between 0 and 1."""
        model.eval()  # Set model to evaluation mode
        criterion = nn.MSELoss()
        total_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()

        mse = total_loss / len(data_loader)  # Average MSE over all batches

        # Scale MSE to be between 0 and 1.  This requires knowing the possible range of your SOC values.
        # Method 1: If you know the min and max SOC in your data:
        min_soc = scaler.data_min_[0] #Get min and max values from scaler
        max_soc = scaler.data_max_[0]
        scaled_mse = mse / ((max_soc - min_soc)**2) #Scale by the range of the data squared

        # Method 2: If you want to scale based on the variance of the target variable:
        # scaled_mse = mse / np.var(scaler.inverse_transform(batch_y.cpu().numpy().reshape(-1,1))) #Scale by the variance of the target variable

        return scaled_mse
    
if __name__ == "__main__":
    
    # Validated RawDataPlotter 3-10-25 6:32 PM
    """
    plotter = RawDataPlotter(raw_data_path, processed_data_dir)
    plotter.plot_columns()
    """

    # not validated yet
    preprocessor = DataPreprocessor()
    # Test 1: Load Data
    df = preprocessor.load_data(raw_data_path)
    if df is not None:
        print(df.head())

    # Test 2: Clean Data
    cleaned_df = preprocessor.clean_data(df)
    print(cleaned_df.head())


    """
    # Preprocess data *before* training the model
    scaled_data, scalers = preprocess_data(df)
    time_step = 60
    X, y = prepare_lstm_data(scaled_data, time_step)
    dataset = TimeSeriesDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = LSTMModel(input_size=1, hidden_size=50, output_size=1)
    model, train_losses = train_model(model, train_loader, num_epochs=50)

    # Plot the training loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{Path('C:/Users/jmani/Documents/BatteryMLProject/src/data').stem}_training_loss.png") #Save figure
    plt.show()

    plot_charge_test_predictions(df, model, scalers, time_step)  # Pass the trained model and scalers

    # Calculate and print scaled MSE
    dataset_eval = TimeSeriesDataset(X, y) #Dataset for evaluation
    eval_loader = DataLoader(dataset_eval, batch_size=32, shuffle=False) #Dataloader for evaluation
    scaled_mse = calculate_scaled_mse(model, eval_loader, scalers[1]) #Calculate scaled MSE, pass in the correct scaler
    print(f"Scaled MSE (0-1): {scaled_mse:.4f}")
    
    plt.close()
    """