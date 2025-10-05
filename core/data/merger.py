import pandas as pd
import glob
import os

# Get all weather_data CSV files
weather_files = glob.glob("weather_data*.csv")

# Read and merge all weather data files
dataframes = []
for file in weather_files:
    df = pd.read_csv(file)
    # Ignore the first column (index 0)
    df = df.iloc[:, 1:]
    dataframes.append(df)

# Merge all dataframes
if dataframes:
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Output to processed-step-1.csv
    merged_df.to_csv("processed-step-1.csv", index=False)
    print(f"Merged {len(dataframes)} files into processed-step-1.csv")
else:
    print("No weather_data files found")