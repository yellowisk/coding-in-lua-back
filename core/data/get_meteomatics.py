import datetime as dt
import meteomatics.api as api
from pandas import DataFrame

username = 'teixeira_giulliano'
password = '02h1ECkoX798Vql374Rp'

def get_data(coordinates: list, start_date: dt.datetime, end_date: dt.datetime):
    model = 'mix'
    startdate = start_date
    enddate = end_date
    interval = dt.timedelta(hours = 24)

    results = []

    parameters = ['t_0m:C', 'max_individual_wave_height:m','mean_wave_direction:d','mean_period_total_swell:s', 'effective_cloud_cover:octas', 'ocean_depth:m']
    q_result: DataFrame = api.query_time_series(coordinates, startdate, enddate, interval, parameters, username, password, model=model)

    for r in q_result.itertuples():
        results.append({
                'lat': float(r.Index[0]),
                'lon': float(r.Index[1]),
                'date': r.Index[2],
                'temperature': float(r[1]),
                'max_individual_wave_height': float(r[2]),
                'mean_wave_direction': float(r[3]),
                'mean_period_total_swell': float(r[4]),
                'clouds': float(r[5]),
                'ocean_depth': float(r[6])
            })

    return results

if __name__ == "__main__":
    coords = [(34.0, -120.0), (30.0, -130.0)]
    start = dt.datetime(2023, 1, 1)
    end = dt.datetime(2023, 1, 10)
    data = get_data(coords, start, end)
    for entry in data:
        print(entry)