from pydantic import BaseModel
from typing import List, Optional

class DataExample(BaseModel):
    id: int
    name: str
    values: List[float]
    description: Optional[str] = None
    
class ClassifierData(BaseModel):
    longitude: float
    latitude: float
    count: int