from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from . import auth, database, models
from app.schemas import User, Token
from app.routes import predict, feedback, train
from sqlalchemy.orm import Session
import uvicorn
from app.database import engine
from app import models

models.Base.metadata.create_all(bind=engine)

app = FastAPI()
app.include_router(auth.router)      # ✅ JWT + Basic auth routes
app.include_router(predict.router)   # 🧠 prediction
app.include_router(feedback.router)  # 💬 feedback
app.include_router(train.router)     # 🧠 train
if __name__ == "__main__":
    uvicorn.run("app.main:app", reload=True)