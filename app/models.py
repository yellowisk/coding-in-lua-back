from pydantic import BaseModel
from typing import List, Optional
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
    coords: List[List[float]]
    date: datetime
    # Optional deltas allow clients to omit them for simulated or simple requests
    deltas: Optional[DeltaGroup] = None
    # Optional view and depth fields for compatibility with earlier clients
    view: Optional[str] = None
    depth: Optional[int] = None