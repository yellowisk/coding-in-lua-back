from pydantic import BaseModel, root_validator
from typing import List, Optional, Union, Dict, Any
from datetime import datetime

class DataExample(BaseModel):
    id: int
    name: str
    values: List[float]
    description: Optional[str] = None
    
class SpotData(BaseModel):
    longitude: float
    latitude: float
    count: int

class DeltaGroup(BaseModel):
    deltaTemp: float
    deltaClouds: float
    deltaOceanDepth: float
    deltaPhytoplankton: float

class ClassifierDataResponse(BaseModel):
    data: List[SpotData]
    
class ClassifierDataRequest(BaseModel):
    # accept coords as list of [lon, lat] or list of dicts {longitude, latitude, phyto?}
    coords: List[Union[List[float], Dict[str, Any]]]
    date: datetime
    deltas: Optional[DeltaGroup] = None
    view: Optional[str] = None
    depth: Optional[int] = None

    @root_validator(pre=True)
    def normalize(cls, values):
        # accept alternative top-level key 'coordinates'
        if 'coords' not in values and 'coordinates' in values:
            values['coords'] = values.pop('coordinates')

        # If coords is provided as a dict with keys, try to convert
        c = values.get('coords')
        if isinstance(c, dict):
            # convert dict-of-coords to list
            # e.g. {"a": [-120,34], "b": [-130,30]} -> [[-120,34], [-130,30]]
            values['coords'] = list(c.values())

        return values