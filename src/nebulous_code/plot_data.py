import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Directory of .csv files
directory = "C:/Users/jmani/Documents/BatteryMLProject/src/data/BMS_Data/"

# List of all CSV files in the directory
csv_files = list(Path(directory).glob("*.csv"))

# Create figure for each CSV file
for csv_file in csv_files:
    # Read CSV file
    df = pd.read_csv(csv_file)

    # Function to handle both EST and EDT timestamps
    def convert_timezone(time_str):
        try:
            return pd.to_datetime(time_str, format='%a %b %d %H:%M:%S EST %Y')
        except:
            return pd.to_datetime(time_str, format='%a %b %d %H:%M:%S EDT %Y')

    # Convert time column
    df['Time'] = df['Time'].apply(convert_timezone)

    # Create new figure for each file
    plt.figure(figsize=(10, 6))

    # Plot data
    plt.scatter(df['Time'], df["Pack State of Charge (SOC)"], alpha=0.5)
    plt.xlabel("Time")
    plt.ylabel("State of Charge (SOC)")
    plt.grid(True)

    # Save figure
    plt.savefig(f"{csv_file.stem}.png")

    # Display the plot
    # plt.show()

    # Close the plot
    plt.close()

""" df = pd.read_csv("C:/Users/jmani/Documents/BatteryMLProject/src/data/BMS_Data/orionbms_log_2024-03-07-09-47-31.csv")

plt.scatter(df["Pack Voltage"], df["Pack State of Charge (SOC)"], alpha = 0.5)
plt.xlabel("Voltage")
plt.ylabel("SOC")
plt.title("Voltage vs SOC")
plt.show()

plt.figure(figsize=(10, 6))

# Scatterplot using seaborn with regression line
sns.regplot(x="Pack Voltage", 
            y="Pack State of Charge (SOC)", 
            data=df,
            scatter_kws={'alpha':0.5},
            line_kws={'color': 'red'})

# Correlation coefficient
correlation = df["Pack Voltage"].corr(df["Pack State of Charge (SOC)"])

# Add correlation coefficient to plot
plt.title(f"Voltage vs SOC\nPearson Correlation: {correlation:.3f}")
plt.xlabel("Voltage")
plt.ylabel("SOC")

# Grid for readability
plt.grid(True, linestyle='--', alpha=0.7)

plt.show()

# Detailed correlation information
print("\nDetailed Statistical Analysis:")
print(sns.lmplot(x="Pack Voltage", 
                 y="Pack State of Charge (SOC)", 
                 data=df).fig) """