from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import example
try:
    from app.routes import classifier
except Exception:
    classifier = None

app = FastAPI()

# Configure CORS to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(example.router, prefix="/examples", tags=["example"])
app.include_router(classifier.router, prefix="/classifier", tags=["classifier"])


@app.get("/")
def read_root():
    return {"Hello": "World"}