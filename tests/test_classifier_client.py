import sys
import os
from fastapi.testclient import TestClient

# Ensure project root is on sys.path so `import main` works when running the script
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import app

client = TestClient(app)

def test_classify_simulated():
    payload = {
        "coords": [[-120.0, 34.0], [-130.0, 30.0]],
        "view": "map",
        "date": "2023-01-05T00:00:00",
        "depth": 50
    }
    resp = client.post('/classifier/classify', json=payload)
    print('status', resp.status_code)
    try:
        print('json', resp.json())
    except Exception:
        print('text', resp.text)

if __name__ == '__main__':
    test_classify_simulated()
