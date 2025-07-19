from fastapi import APIRouter, Depends
from pydantic import BaseModel
from backend.app.auth import get_current_user
from backend.app.llm_advice import get_llm_advice

router = APIRouter()

class AdviceRequest(BaseModel):
    prediction: str
    patient_data: dict

@router.post("/advise")
def advise(data: AdviceRequest, user: str = Depends(get_current_user)):
    advice = get_llm_advice(data.prediction, data.patient_data)
    return {"advice": advice}
