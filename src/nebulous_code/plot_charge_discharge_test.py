import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Directory where live test data is stored
directory = "C:/Users/jmani/Documents/BatteryMLProject/src/data/BMS_Data/live_tests/orionbms_log_2025-02-07-21-19-32testdata1.csv"

# Read CSV file
df = pd.read_csv(directory)


# Convert time column to handle EST timestamps
df['Time'] = pd.to_datetime(df['Time'], format='%a %b %d %H:%M:%S EST %Y')

# Create new figure
plt.figure(figsize=(10, 6))

# Plot data
plt.scatter(df['Time'], df["Pack State of Charge (SOC)"], alpha=0.5)
plt.xlabel("Time")
plt.ylabel("State of Charge (SOC)")
plt.grid(True)

# Save figure
plt.savefig(f"{Path('C:/Users/jmani/Documents/BatteryMLProject/src/data').stem}.png")

plt.show
plt.close()