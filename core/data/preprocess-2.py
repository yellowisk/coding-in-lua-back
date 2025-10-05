import pandas as pd
import os
import bisect

def find_phyto_file_for_date(date, phyto_folder):
    """
    Find the appropriate phytoplankton file based on the date.
    Files are named like: AQUA_MODIS.YYYYMMDD_YYYYMMDD.L3m.MO.CHL.x_chlor_a.csv
    """
    year = date.year
    month = date.month
    
    # Find files for the specific year and month
    files = os.listdir(phyto_folder)
    for file in files:
        if not file.endswith('.csv'):
            continue
        
        # Extract date from filename: AQUA_MODIS.20200101_20200131...
        parts = file.split('.')
        if len(parts) < 2:
            continue
        
        date_range = parts[1]  # e.g., "20200101_20200131"
        start_date_str = date_range.split('_')[0]  # "20200101"
        
        try:
            file_year = int(start_date_str[:4])
            file_month = int(start_date_str[4:6])
            
            if file_year == year and file_month == month:
                return os.path.join(phyto_folder, file)
        except:
            continue
    
    return None

def binary_search_closest(arr, target):
    """
    Binary search to find the index of the closest value to target in a sorted array.
    Returns the index of the closest value.
    """
    if len(arr) == 0:
        return None
    
    # Use bisect to find insertion point
    pos = bisect.bisect_left(arr, target)
    
    if pos == 0:
        return 0
    if pos == len(arr):
        return len(arr) - 1
    
    # Compare with neighbors to find closest
    before = arr[pos - 1]
    after = arr[pos]
    
    if abs(target - before) < abs(target - after):
        return pos - 1
    else:
        return pos

def load_phyto_data(file_path):
    """
    Load phytoplankton data and prepare it for binary search.
    Returns sorted unique coordinates and the full dataframe.
    """
    try:
        df = pd.read_csv(file_path)
        
        # Convert to numeric
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['chlor_a'] = pd.to_numeric(df['chlor_a'], errors='coerce')
        
        # Remove NaN values
        df = df.dropna()
        
        # Get unique sorted coordinates
        unique_lons = sorted(df['lon'].unique())
        unique_lats = sorted(df['lat'].unique())
        
        return df, unique_lons, unique_lats
    except Exception as e:
        print(f"Error loading phyto data from {file_path}: {e}")
        return None, None, None

def find_chlorophyll(phyto_df, phyto_lons, phyto_lats, target_lon, target_lat):
    """
    Find the chlorophyll value for the closest coordinates using binary search.
    """
    if phyto_df is None or len(phyto_df) == 0:
        return None
    
    # Find closest longitude
    lon_idx = binary_search_closest(phyto_lons, target_lon)
    if lon_idx is None:
        return None
    closest_lon = phyto_lons[lon_idx]
    
    # Filter dataframe to rows with this longitude
    df_lon = phyto_df[phyto_df['lon'] == closest_lon]
    
    if len(df_lon) == 0:
        return None
    
    # Get sorted latitudes for this longitude
    lats_for_lon = sorted(df_lon['lat'].unique())
    
    # Find closest latitude
    lat_idx = binary_search_closest(lats_for_lon, target_lat)
    if lat_idx is None:
        return None
    closest_lat = lats_for_lon[lat_idx]
    
    # Get the chlorophyll value
    result = df_lon[df_lon['lat'] == closest_lat]['chlor_a'].values
    
    if len(result) > 0:
        return result[0]
    
    return None

def main():
    # Paths
    input_csv = '/home/gvinfinity/SpaceAppsChallenge/coding-in-lua-back/core/data/processed-step-1.csv'
    phyto_folder = '/home/gvinfinity/SpaceAppsChallenge/new_phyto/Processed'
    output_csv = '/home/gvinfinity/SpaceAppsChallenge/coding-in-lua-back/core/data/processed-step-2.csv'
    
    # Load the main dataset
    print("Loading processed-step-1.csv...")
    df = pd.read_csv(input_csv)
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Add chlorophyll column
    df['phyto'] = None
    
    # Cache for phytoplankton data by file
    phyto_cache = {}
    
    print(f"Processing {len(df)} rows...")
    
    # Process each row
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Processing row {idx}/{len(df)}...")
        
        date = row['date']
        lon = row['lon']
        lat = row['lat']
        
        # Find the appropriate phyto file for this date
        phyto_file = find_phyto_file_for_date(date, phyto_folder)
        
        if phyto_file is None:
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
        
        if chlor_value is not None:
            df.at[idx, 'phyto'] = chlor_value
    
    # Save the result
    print(f"Saving results to {output_csv}...")
    df.to_csv(output_csv, index=False)
    
    # Print statistics
    total_rows = len(df)
    rows_with_chlor = df['phyto'].notna().sum()
    print(f"\nProcessing complete!")
    print(f"Total rows: {total_rows}")
    print(f"Rows with chlorophyll data: {rows_with_chlor}")
    print(f"Success rate: {rows_with_chlor/total_rows*100:.2f}%")

if __name__ == "__main__":
    main()
