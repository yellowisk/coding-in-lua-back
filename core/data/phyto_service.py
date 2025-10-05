import os
from typing import List, Tuple, Dict, Optional
from collections import OrderedDict
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
_env_folder = os.environ.get('PHYTO_FOLDER')
CANDIDATE_FOLDERS = [p for p in [
    _env_folder,
    os.path.join(REPO_ROOT, 'phyto', 'Processed'),
    os.path.join(REPO_ROOT, 'phyto'),
    os.path.join(BASE_DIR, 'phyto_data'),
] if p]

# visible folder chosen (first that contains files)
PHYTO_FOLDER = None

# index YYYYMM -> filepath
_YYYYMM_INDEX: Dict[str, str] = {}

# loaded file cache: path -> dict with numpy arrays 'lon','lat','chla'
_LOADED_FILES: Dict[str, Dict[str, np.ndarray]] = {}

# LRU per-point cache: (path, rlon, rlat) -> chla
_PHYTO_POINT_CACHE_MAX = int(os.environ.get('PHYTO_POINT_CACHE_MAX', '10000'))
_PHYTO_POINT_CACHE: "OrderedDict[Tuple[str,float,float], Optional[float]]" = OrderedDict()

# search radius in degrees for initial neighborhood filter (can be widened)
_PHYTO_SEARCH_RADIUS = float(os.environ.get('PHYTO_SEARCH_RADIUS', '0.5'))


def _build_index_once():
    global PHYTO_FOLDER
    for folder in CANDIDATE_FOLDERS:
        try:
            if not os.path.isdir(folder):
                continue
            files = [f for f in os.listdir(folder) if f.lower().endswith('.csv')]
            if not files:
                continue
            # prefer this folder if it contains files
            if PHYTO_FOLDER is None:
                PHYTO_FOLDER = folder
            for fn in files:
                name = fn
                for i in range(len(name) - 5):
                    seg = name[i:i+6]
                    if seg.isdigit():
                        yyyy = seg[:4]
                        mm = seg[4:6]
                        try:
                            im = int(mm)
                            iy = int(yyyy)
                            if 1 <= im <= 12:
                                key = f"{iy:04d}{im:02d}"
                                if key not in _YYYYMM_INDEX:
                                    _YYYYMM_INDEX[key] = os.path.join(folder, fn)
                                break
                        except Exception:
                            continue
        except Exception:
            continue


_build_index_once()


def _phyto_point_cache_get(key: Tuple[str, float, float]) -> Optional[float]:
    try:
        val = _PHYTO_POINT_CACHE.pop(key)
        _PHYTO_POINT_CACHE[key] = val
        return val
    except KeyError:
        return None


def _phyto_point_cache_set(key: Tuple[str, float, float], value: Optional[float]):
    if key in _PHYTO_POINT_CACHE:
        _PHYTO_POINT_CACHE.pop(key, None)
    _PHYTO_POINT_CACHE[key] = value
    try:
        while len(_PHYTO_POINT_CACHE) > _PHYTO_POINT_CACHE_MAX:
            _PHYTO_POINT_CACHE.popitem(last=False)
    except Exception:
        pass


def _load_file_to_arrays(path: str) -> Optional[Dict[str, np.ndarray]]:
    # return cached arrays if present
    if path in _LOADED_FILES:
        return _LOADED_FILES[path]
    try:
        df = pd.read_csv(path)
        if 'chlor_a' not in df.columns and 'chl' in df.columns:
            df['chlor_a'] = df['chl']
        df = df.rename(columns={c: c.strip() for c in df.columns})
        if not set(['lon', 'lat', 'chlor_a']).issubset(df.columns):
            return None
        sdf = df[['lon', 'lat', 'chlor_a']].dropna()
        sdf['lon'] = pd.to_numeric(sdf['lon'], errors='coerce')
        sdf['lat'] = pd.to_numeric(sdf['lat'], errors='coerce')
        sdf['chlor_a'] = pd.to_numeric(sdf['chlor_a'], errors='coerce')
        sdf = sdf.dropna(subset=['lon', 'lat', 'chlor_a'])
        lons = sdf['lon'].to_numpy()
        lats = sdf['lat'].to_numpy()
        chla = sdf['chlor_a'].to_numpy()
        arr = {'lon': lons, 'lat': lats, 'chla': chla}
        _LOADED_FILES[path] = arr
        return arr
    except Exception:
        return None


def _find_path_for_date(date) -> Optional[str]:
    yrmon = f"{date.year:04d}{date.month:02d}"
    path = _YYYYMM_INDEX.get(yrmon)
    if path:
        return path
    # fallback: try to scan candidate folders (rare)
    for folder in CANDIDATE_FOLDERS:
        try:
            if not os.path.isdir(folder):
                continue
            for fn in os.listdir(folder):
                if yrmon in fn and fn.lower().endswith('.csv'):
                    p = os.path.join(folder, fn)
                    _YYYYMM_INDEX[yrmon] = p
                    return p
        except Exception:
            continue
    return None


def _nearest_in_arrays(arr: Dict[str, np.ndarray], lon: float, lat: float, radius: float) -> Optional[float]:
    lons = arr['lon']
    lats = arr['lat']
    chla = arr['chla']
    # boolean mask for bounding box
    mask = (np.abs(lons - lon) <= radius) & (np.abs(lats - lat) <= radius)
    if not np.any(mask):
        return None
    idxs = np.nonzero(mask)[0]
    # compute squared distances
    dx = lons[idxs] - lon
    dy = lats[idxs] - lat
    d2 = dx*dx + dy*dy
    i = int(np.argmin(d2))
    return float(chla[idxs[i]])


def get_phyto_for_coords(date, coords: List[Tuple[float, float]]) -> Dict[Tuple[float, float], Optional[float]]:
    """Return mapping (rounded lon, rounded lat) -> chlor_a or None.

    Strategy:
    - choose a single CSV for the requested date (YYYYMM) using the index
    - load that file (cached) into numpy arrays
    - for each coordinate, check per-point LRU cache first
    - then search nearby rows within a small radius using vectorized numpy masks
    - if no candidate found, attempt a widened radius before giving up
    """
    out: Dict[Tuple[float, float], Optional[float]] = {}
    path = _find_path_for_date(date)
    if not path:
        # no file found for the month
        for lon, lat in coords:
            out[(round(float(lon), 5), round(float(lat), 5))] = None
        return out

    arr = _load_file_to_arrays(path)
    if arr is None:
        for lon, lat in coords:
            out[(round(float(lon), 5), round(float(lat), 5))] = None
        return out

    for lon, lat in coords:
        rlon = round(float(lon), 5)
        rlat = round(float(lat), 5)
        key = (path, rlon, rlat)
        cached = _phyto_point_cache_get(key)
        if cached is not None:
            out[(rlon, rlat)] = cached
            continue

        # try initial radius
        val = _nearest_in_arrays(arr, float(lon), float(lat), _PHYTO_SEARCH_RADIUS)
        if val is None:
            # try wider radius before giving up
            val = _nearest_in_arrays(arr, float(lon), float(lat), max(2.0, _PHYTO_SEARCH_RADIUS * 4))
        _phyto_point_cache_set(key, val)
        out[(rlon, rlat)] = val
    return out


# public alias
get_phyto_for_coords.__doc__ = "Return mapping (rounded lon, rounded lat) -> chlor_a or None using optimized service"
