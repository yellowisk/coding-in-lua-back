import pandas as pd
import numpy as np
import datetime as dt
import os
import sys
from get_meteomatics import get_data

# Import functions from preprocess-2
sys.path.append('/home/gvinfinity/SpaceAppsChallenge/coding-in-lua-back/core/data')
from preprocess_2_helpers import find_phyto_file_for_date, load_phyto_data, find_chlorophyll

def get_date_range_from_df(df):
    """Extract the date range from the dataframe"""
    df['date'] = pd.to_datetime(df['date'])
    start_date = df['date'].min()
    end_date = df['date'].max()
    return start_date, end_date

def generate_negative_samples(n_coordinates, lat_range=(-70, 70), lon_range=(-100, 25)):
    """Generate uniformly distributed longitude and latitude points"""
    print(f"Generating {n_coordinates} coordinates...")
    
    # Generate random points
    lats = np.random.uniform(lat_range[0], lat_range[1], n_coordinates)
    lons = np.random.uniform(lon_range[0], lon_range[1], n_coordinates)

    return list(zip(lats, lons))

def get_weather_and_phyto_data(coordinates, start_date, end_date, phyto_folder):
    """
    Get weather data from meteomatics API and phyto data for given coordinates
    coordinates: list of tuples [(lat1, lon1), (lat2, lon2), ...]
    Returns a list of dictionaries with all the data
    """
    print(f"Fetching weather data for {len(coordinates)} coordinates...")
    
    # Get weather data for all coordinates in a single API call
    # Note: coordinates format for meteomatics is [(lat, lon), ...]
    weather_results = get_data(coordinates, start_date, end_date)
    
    # Randomly select 3 days worth of data for each coordinate
    print("Selecting 3 random days for each coordinate...")
    filtered_results = []

    # Group results by coordinate
    coordinate_groups = {}
    for entry in weather_results:
        coord_key = (entry['lat'], entry['lon'])
        if coord_key not in coordinate_groups:
            coordinate_groups[coord_key] = []
        coordinate_groups[coord_key].append(entry)

    # For each coordinate, randomly select 3 entries
    for coord_key, entries in coordinate_groups.items():
        if len(entries) >= 3:
            selected_entries = np.random.choice(entries, size=3, replace=False)
            filtered_results.extend(selected_entries)
        else:
            # If less than 3 days available, take all available
            filtered_results.extend(entries)

    weather_results = filtered_results
    
    print(f"Received {len(weather_results)} weather data entries")
    print("Adding phytoplankton data...")
    
    # Cache for phyto data by file
    phyto_cache = {}
    
    # Add phytoplankton data to each result
    for idx, entry in enumerate(weather_results):
        if idx % 100 == 0:
            print(f"Processing phyto data {idx}/{len(weather_results)}...")
        
        date = pd.to_datetime(entry['date'])
        lat = entry['lat']
        lon = entry['lon']
        
        # Find the appropriate phyto file for this date
        phyto_file = find_phyto_file_for_date(date, phyto_folder)
        
        if phyto_file is None:
            entry['phyto'] = None
            continue
        
        # Load phyto data (use cache if available)
        if phyto_file not in phyto_cache:
            print(f"Loading phytoplankton data from {os.path.basename(phyto_file)}...")
            phyto_df, phyto_lons, phyto_lats = load_phyto_data(phyto_file)
            phyto_cache[phyto_file] = (phyto_df, phyto_lons, phyto_lats)
        else:
            phyto_df, phyto_lons, phyto_lats = phyto_cache[phyto_file]
        
        # Find chlorophyll value
        chlor_value = find_chlorophyll(phyto_df, phyto_lons, phyto_lats, lon, lat)
        entry['phyto'] = chlor_value
    
    return weather_results

def main():
    # Paths
    input_csv = '/home/gvinfinity/SpaceAppsChallenge/coding-in-lua-back/core/data/processed-step-2.csv'
    phyto_folder = '/home/gvinfinity/SpaceAppsChallenge/new_phyto/Processed'
    output_csv = '/home/gvinfinity/SpaceAppsChallenge/coding-in-lua-back/core/data/processed-step-3-test.csv'
    
    # Step 1: Load processed-step-2.csv (use only first 10 rows for testing)
    print("Loading processed-step-2.csv (test mode - first 10 rows)...")
    df_sharks = pd.read_csv(input_csv, nrows=10)
    
    # Add shark column with value 1 for all rows (these are shark observations)
    df_sharks['shark'] = 1
    
    print(f"Loaded {len(df_sharks)} shark observations")
    
    # Step 2: Get date range from the data
    start_date, end_date = get_date_range_from_df(df_sharks)
    print(f"Date range: {start_date} to {end_date}")
    
    # Step 3: Generate negative samples (no shark observations)
    # Generate same number of negative samples as positive samples
    n_negative_samples = int(len(df_sharks) / 3)

    coordinates = generate_negative_samples(n_negative_samples)
    
    
    # Step 4: Get weather and phyto data for negative samples in a single batch
    print("\nFetching data for negative samples...")
    negative_data = get_weather_and_phyto_data(
        coordinates, 
        start_date.to_pydatetime(), 
        end_date.to_pydatetime(),
        phyto_folder
    )
    
    # Step 5: Create dataframe from negative samples
    print("\nCreating negative samples dataframe...")
    df_negative = pd.DataFrame(negative_data)
    df_negative['shark'] = 0
    
    print(f"Created {len(df_negative)} negative sample entries")
    
    # Step 6: Merge both dataframes
    print("\nMerging dataframes...")
    df_final = pd.concat([df_sharks, df_negative], ignore_index=True)
    
    # Replace NaN and -666 values with 0
    print("Replacing NaN and -666 values with 0...")
    df_final = df_final.fillna(0)
    df_final = df_final.replace(-666, 0)
    
    # Shuffle the data
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Final dataset size: {len(df_final)} rows")
    print(f"Shark observations: {(df_final['shark'] == 1).sum()}")
    print(f"Non-shark observations: {(df_final['shark'] == 0).sum()}")
    
    # Step 7: Save to CSV
    print(f"\nSaving to {output_csv}...")
    df_final.to_csv(output_csv, index=False)
    
    print("\nProcessing complete!")
    print(f"Output saved to: {output_csv}")
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total rows: {len(df_final)}")
    print(f"Columns: {list(df_final.columns)}")
    print(f"Missing values per column:")
    print(df_final.isnull().sum())
    
    # Show sample of data
    print("\nSample of final data:")
    print(df_final.head(10))

if __name__ == "__main__":
    main()
