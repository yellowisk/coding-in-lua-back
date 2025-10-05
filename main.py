from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

if classifier is not None and hasattr(classifier, 'router'):
    app.include_router(classifier.router, prefix="/classifier", tags=["classifier"])
else:
    print("Warning: Classifier router could not be included.")

@app.get("/")
def read_root():
    return {"Hello": "World"}