import os
from typing import List, Tuple, Dict, Optional

# Backwards-compatible wrapper that delegates to the optimized phyto service.
try:
    from core.data.phyto_service import get_phyto_for_coords as _service_get_phyto_for_coords
except Exception:
    _service_get_phyto_for_coords = None


def get_phyto_for_coords(date, coords: List[Tuple[float, float]]) -> Dict[Tuple[float, float], Optional[float]]:
    """Compatibility wrapper that delegates to the optimized phyto_service.

    If the service isn't available, falls back to returning None values.
    """
    if _service_get_phyto_for_coords is None:
        return { (round(float(lon),5), round(float(lat),5)): None for lon, lat in coords }
    return _service_get_phyto_for_coords(date, coords)
