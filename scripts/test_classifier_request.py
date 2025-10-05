import requests
import json

URL = 'http://127.0.0.1:8000/classifier/classify'

payload = {
    "coords": [[-120.0, 34.0], [-130.0, 30.0]],
    "view": "map",
    "date": "2023-01-05T00:00:00",
    "depth": 50
}

resp = requests.post(URL, json=payload, timeout=10)
print('status', resp.status_code)
try:
    print(json.dumps(resp.json(), indent=2))
except Exception:
    print(resp.text)
