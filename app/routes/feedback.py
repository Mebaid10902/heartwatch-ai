from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.schemas import FeedbackInput
from app.auth import get_current_user
from app.models import Feedback
from app.dependencies import get_db
from app.database import SessionLocal

router = APIRouter()
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
@router.post("/feedback")
def submit_feedback(input_data: FeedbackInput, db: Session = Depends(get_db), user: str = Depends(get_current_user)):
    feedback = Feedback(
        age=input_data.age,
        sex=input_data.sex,
        cp=input_data.cp,
        trestbps=input_data.trestbps,
        chol=input_data.chol,
        fbs=input_data.fbs,
        restecg=input_data.restecg,
        thalch=input_data.thalch,
        exang=input_data.exang,
        oldpeak=input_data.oldpeak,
        slope=input_data.slope,
        ca=input_data.ca,
        thal=input_data.thal,
        prediction = input_data.prediction
        
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return {"msg": "✅ Feedback stored", "id": feedback.id,"prediction": feedback.prediction, "user": user}
