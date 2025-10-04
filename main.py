from fastapi import FastAPI
from app.routes import example

app = FastAPI()

app.include_router(example.router, prefix="/examples", tags=["example"])

@app.get("/")
def read_root():
    return {"Hello": "World"}