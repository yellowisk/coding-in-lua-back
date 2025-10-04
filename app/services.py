from fastapi import APIRouter
from typing import List
from app.models import DataExample

router = APIRouter()

@router.get("/dataexamples/", response_model=List[DataExample])
def read_dataexamples():
    return [
        DataExample(id=1, name="Example 1", values=[1.0, 2.0, 3.0], description="First example"),
        DataExample(id=2, name="Example 2", values=[4.0, 5.0], description=None),
    ]