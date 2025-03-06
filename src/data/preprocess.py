import pandas as pd
import numpy as np
import glob
import os
from statistical_analysi import stat_extractor

input_directory = "C:/Users/jmani/Documents/BatteryMLProject/src/data/complete_dataset/raw_data/*.csv"
output_directory = "C:/Users/jmani/Documents/BatteryMLProject/src/data/complete_dataset/processed_data"

def load_data(file_path):
    # Load raw data from a CSV file.
    return pd.read_csv(file_path)

def clean_data(df):
    """Cleans the battery dataset by handling time, missing values, and outliers."""

    # Convert time column to handle EST timestamps
    try:
        df['Time'] = pd.to_datetime(df['Time'], format='%a %b %d %H:%M:%S EST %Y')
    except Exception as e:
        print(f"Error converting time column: {e}")

    # Drop rows with invalid time entries
    df.dropna(subset=['Time'], inplace=True)
    
    # Handle missing values (forward fill then backward fill)
    df = df.ffill().bfill()

    # Remove outliers using IQR method
    def remove_outliers_iqr(df, cols):
        for col in cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            

        return df[(df[cols] >= lower_bound) & (df[cols] <= upper_bound)]
    
    # Apply outlier removal for numeric columns (excluding 'Time')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = remove_outliers_iqr(df, numeric_cols)
    

    return df

def preprocess(input_directory, output_directory):
    """Processes all CSV files in the input directory and saves them to the output directory."""
    file_list = glob.glob(input_directory)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for file in file_list:
        # Load raw data
        df = load_data(file)

        # Clean data
        df = clean_data(df)

        # Generate output file path
        output_file = os.path.join(output_directory, os.path.basename(file))

        # Save processed data
        df.to_csv(output_file, index=False)

        print(f"Processed file saved: {output_file}")

if __name__ == "__main__":
    preprocess(input_directory, output_directory)
    print("Data preprocessing complete.")