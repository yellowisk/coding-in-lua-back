import json
from datetime import datetime
import folium

with open('/home/gvinfinity/SpaceAppsChallenge/coding-in-lua-back/core/data/sharkpulse.json', 'r') as f:
    data = json.load(f)
    for d in data:
        if not d['date']:
            continue
        d['date'] = datetime.strptime(d['date'], '%a, %d %b %Y %H:%M:%S GMT')

    data = list(filter(lambda x: x['date'] and x['date'] > datetime(2020, 1, 1) and -70 <= float(x['latitude']) <= 70 and -100 <= float(x['longitude']) <= 25, data))

    sorted_data = sorted(data, key=lambda x: x['date'])

    # Create a map centered on a reasonable location
    map_center = [0, 0]  # Default center, will be adjusted based on data
    if sorted_data:
        # Calculate center based on data points
        avg_lat = sum(float(d.get('latitude', 0)) for d in sorted_data if d.get('latitude')) / len([d for d in sorted_data if d.get('latitude')])
        avg_lon = sum(float(d.get('longitude', 0)) for d in sorted_data if d.get('longitude')) / len([d for d in sorted_data if d.get('longitude')])
        map_center = [avg_lat, avg_lon]

    # Create the map
    shark_map = folium.Map(location=map_center, zoom_start=5)

    # Add pins for each data point
    for d in sorted_data:
        if d.get('latitude') and d.get('longitude'):
            popup_text = f"Date: {d['date']}<br>Location: ({d['latitude']}, {d['longitude']})"
            folium.Marker(
                location=[float(d['latitude']), float(d['longitude'])],
                popup=popup_text,
                tooltip="Shark sighting"
            ).add_to(shark_map)

    # Save the map
    shark_map.save('map.html')
    print(f"Map saved with {len(sorted_data)} data points")
    print(*sorted_data[-10:], sep='\n')