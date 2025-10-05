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

class ClassifierDataResponse(BaseModel):
    data: List[SpotData]
    
class ClassifierDataRequest(BaseModel):
    coordinates: List[List[float]]
    view: str
    date: datetime
    depth: int