from fastapi import APIRouter
import subprocess
from fastapi import APIRouter, Depends
from app.auth import get_current_user 
router = APIRouter()

@router.post("/retrain")
def retrain(user: str = Depends(get_current_user)):
    subprocess.run(["python", "train_model.py"])
    return {"message": "Model retrained and saved."}