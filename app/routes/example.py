from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def list_examples():
    return [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
    ]

@router.get("/{user_id}")
def get_example(user_id: int):
    return {"id": user_id, "name": f"User {user_id}"}