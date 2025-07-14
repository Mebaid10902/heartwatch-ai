# 💓 Heart Disease Prediction App (FastAPI + Streamlit + ML)

A full-stack machine learning web app for predicting heart disease using FastAPI (backend) and Streamlit (frontend). Includes feedback collection, retraining functionality, JWT-based auth, and Docker support.

## 📦 Features

- 🧠 Machine Learning with Scikit-learn, XGBoost
- 🩺 API prediction endpoint (`/predict`) with JWT auth
- ✍️ Feedback submission stored in SQLite
- 🔁 Retraining API (`/retrain`)
- 🔐 Login using JWT auth
- 📊 Streamlit frontend for prediction and feedback
- 🐳 Dockerized (FastAPI & Streamlit)

---

## 🚀 How to Run (Locally)

### 1. Backend (Model training && FastApi)
```bash
python -m ml.train_model    
```
```bash
uvicorn app.main:app --reload
```
visit http://localhost:8000/docs

### 2. Frontend (Streamlit)
```bash
streamlit run streamlit_app/streamlit_app.py
```
Then visit http://localhost:8501
---

## 🐳 Run with Docker (Recommended)

### 1. Build containers
```bash
 docker buildx build -t heart-app .
```

### 3. Run containers
```bash
docker run -d -p 8000:8000 --name heart-api heart-app
```
### 3. Build Docker compose file
```bash
docker compose up --build
```


Then visit: http://localhost:8501

## 🧪 Run Pytest
```bash
$env:PYTHONPATH = "."
pytest tests/test_api.py

```

---

## ✅ Sample Users
- username: `admin`
- password: `password123`

---
