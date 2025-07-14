from fastapi import APIRouter, Depends, HTTPException
from app.auth import get_current_user
from app.schemas import HeartInput
import numpy as np
import joblib
import os

router = APIRouter()

@router.post("/predict")
def predict(input_data: HeartInput, user: str = Depends(get_current_user)):
    model_path = "models/model.pkl"

    # 🔍 Check model exists
    if not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail="Model file not found.")

    # 🔄 Load model
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    # 🔢 Convert input to NumPy array
    features = np.array([[
        input_data.age,
        input_data.sex,
        input_data.cp,
        input_data.trestbps,
        input_data.chol,
        input_data.fbs,
        input_data.restecg,
        input_data.thalch,
        input_data.exang,
        input_data.oldpeak,
        input_data.slope,
        input_data.ca,
        input_data.thal,
    ]])

    # 🤖 Predict
    try:
        prediction = int(model.predict(features)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # 🧾 Logging (optional for debugging)
    print(f"🧠 User: {user} | Input: {features.tolist()} | Prediction: {prediction}")

    # 📦 Return formatted response
    return {
        "prediction": "💔 Heart Disease" if prediction == 1 else "❤️ No Heart Disease",
        "predicted_class": prediction,
        "user": user
    }
