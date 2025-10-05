import json
from datetime import datetime
from get_meteomatics import get_data
import pandas as pd

with open('/home/gvinfinity/SpaceAppsChallenge/coding-in-lua-back/core/data/sharkpulse.json', 'r') as f:
    data = json.load(f)
    for d in data:
        if not d['date']:
            continue
        d['date'] = datetime.strptime(d['date'], '%a, %d %b %Y %H:%M:%S GMT')

    data = list(filter(lambda x: x['date'] and x['date'] > datetime(2020, 1, 1) and x['date'] < datetime(2025, 1, 1) and -70 <= float(x['latitude']) <= 70 and -100 <= float(x['longitude']) <= 25, data))

    
    df = pd.DataFrame(data)
    grouped_data = df.groupby(df['date'].dt.floor('D'))
    results = []
    i = 0
    for date, group in grouped_data:
        if i < 1990:
            i += 1
            continue
        
        if i >= 3000:
            break
        
        coordinates = [(row['latitude'], row['longitude']) for _, row in group.iterrows()]
        try:
            weather_data = get_data(coordinates, date, date)
        except Exception as e:
            print(f"Error fetching weather data for {date}: {e}")
            df = pd.DataFrame(results)
            saved_file = f"/home/gvinfinity/SpaceAppsChallenge/coding-in-lua-back/core/data/weather_data_{i}second.csv"
            df.to_csv(saved_file, index=True)
            break
        results.extend(weather_data)
        i += 1

    results_df = pd.DataFrame(results)
    saved_file = f"/home/gvinfinity/SpaceAppsChallenge/coding-in-lua-back/core/data/weather_data_1990_.csv"
    results_df.to_csv(saved_file, index=True)
    print(f"Saved weather data to {saved_file}")