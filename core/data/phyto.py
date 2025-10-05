import os
import bisect
from typing import List, Tuple, Dict, Optional
import pandas as pd

# Folder for phyto CSVs (defaults to core/data/phyto_data)
BASE_DIR = os.path.dirname(__file__)
PHYTO_FOLDER = os.environ.get('PHYTO_FOLDER', os.path.join(BASE_DIR, 'phyto_data'))

# Simple in-memory cache for opened files
_PHYTO_CACHE: Dict[str, Tuple[pd.DataFrame, List[float], Dict[float, List[float]]]] = {}


def _find_phyto_file_for_date(date) -> Optional[str]:
    # naive: choose a file that contains YYYYMM in its name
    yrmon = f"{date.year:04d}{date.month:02d}"
    try:
        for fn in os.listdir(PHYTO_FOLDER):
            if yrmon in fn and fn.endswith('.csv'):
                return os.path.join(PHYTO_FOLDER, fn)
    except Exception:
        return None
    return None


def _load_phyto(path: str):
    if path in _PHYTO_CACHE:
        return _PHYTO_CACHE[path]
    try:
        df = pd.read_csv(path)
        # normalize column names
        if 'chlor_a' not in df.columns and 'chl' in df.columns:
            df['chlor_a'] = df['chl']
        df = df.rename(columns={c: c.strip() for c in df.columns})
        df = df[['lon', 'lat', 'chlor_a']].dropna()
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['chlor_a'] = pd.to_numeric(df['chlor_a'], errors='coerce')
        df = df.dropna(subset=['lon', 'lat', 'chlor_a'])
        unique_lons = sorted(df['lon'].unique())
        lon_to_lats = {lon: sorted(df[df['lon'] == lon]['lat'].unique()) for lon in unique_lons}
        _PHYTO_CACHE[path] = (df, unique_lons, lon_to_lats)
        return _PHYTO_CACHE[path]
    except Exception:
        return None, [], {}


def _closest_index(sorted_arr: List[float], value: float) -> Optional[int]:
    if not sorted_arr:
        return None
    import bisect
    pos = bisect.bisect_left(sorted_arr, value)
    if pos == 0:
        return 0
    if pos >= len(sorted_arr):
        return len(sorted_arr) - 1
    before = sorted_arr[pos - 1]
    after = sorted_arr[pos]
    return pos - 1 if abs(value - before) <= abs(value - after) else pos


def _lookup_chlor_a(df: pd.DataFrame, lons: List[float], lon_to_lats: Dict[float, List[float]], lon: float, lat: float) -> Optional[float]:
    li = _closest_index(lons, lon)
    if li is None:
        return None
    closest_lon = lons[li]
    lats = lon_to_lats.get(closest_lon, [])
    if not lats:
        return None
    lati = _closest_index(lats, lat)
    if lati is None:
        return None
    closest_lat = lats[lati]
    vals = df[(df['lon'] == closest_lon) & (df['lat'] == closest_lat)]['chlor_a'].values
    if len(vals) == 0:
        return None
    try:
        return float(vals[0])
    except Exception:
        return None


def get_phyto_for_coords(date, coords: List[Tuple[float, float]]) -> Dict[Tuple[float, float], Optional[float]]:
    """Return dict mapping (lon, lat) rounded to 5 decimals -> chlorophyll (or None).

    Looks for files under `PHYTO_FOLDER` containing YYYYMM in the filename.
    """
    out: Dict[Tuple[float, float], Optional[float]] = {}
    path = _find_phyto_file_for_date(date)
    if not path:
        return { (round(lon,5), round(lat,5)): None for lon, lat in coords }
    df, lons, lon_to_lats = _load_phyto(path)
    if df is None:
        return { (round(lon,5), round(lat,5)): None for lon, lat in coords }
    for lon, lat in coords:
        val = _lookup_chlor_a(df, lons, lon_to_lats, lon, lat)
        out[(round(lon,5), round(lat,5))] = val
    return out
