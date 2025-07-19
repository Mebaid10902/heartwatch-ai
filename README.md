# 💓 Heart Disease Risk Prediction & LLM Advice System

An end-to-end **machine learning backend system** built with **FastAPI**, featuring heart disease prediction, **LLM-powered medical recommendations**, user feedback collection, retraining pipeline, and **secure JWT-authenticated APIs**. Supports optional frontend with Streamlit and full Docker-based deployment.

---

## ⚙️ Features

- 🧠 **ML Models:** Trained using Scikit-learn, XGBoost with automatic versioning
- 🔐 **JWT Auth:** Secure login and token-based access to prediction & feedback APIs
- 📈 **/predict API:** Predicts heart disease from patient features
- 📝 **/feedback API:** Collects user-labeled data for retraining
- 🔁 **/retrain API:** On-demand retraining of models with persisted updates
- 💬 **/advise API:** Uses an **LLM** (e.g. Hugging Face model) to generate personalized lifestyle advice
- 🗄️ **SQLite** as the storage layer for feedback data
- 🧪 Includes **unit tests with Pytest**
- 🐳 Dockerized for production-readiness
- 🎨 Optional **Streamlit** frontend for manual input & visualization

---

## 🚀 Quickstart (Local Development)

### ✅ 1. Train the initial ML model
```bash
python -m ml.train_model
✅ 2. Start the FastAPI Backend
bash
uvicorn app.main:app --reload
Visit Swagger docs: http://localhost:8000/docs

🌐 API Overview
Endpoint	Method	Description	Auth
/login	POST	Get JWT token	❌
/predict	POST	Predict heart disease	✅
/advise	POST	Generate LLM-based lifestyle advice	✅
/feedback	POST	Submit patient data + true label	✅
/retrain	POST	Trigger retraining from feedback	✅

📊 Optional: Run Streamlit Frontend
bash
streamlit run streamlit_app/streamlit_app.py
Then open: http://localhost:8501

🐳 Docker Setup (Recommended)
🛠️ 1. Build Docker container
bash
docker build -t heartwatch-api .
▶️ 2. Run backend container
bash

docker run -d -p 8000:8000 --name heart-api heartwatch-api
🧩 3. With Docker Compose (API + Streamlit)
bash
docker compose up --build
Then visit: http://localhost:8501

🧪 Run Pytest (Unit Tests)
bash
$env:PYTHONPATH = "."   # On Windows PowerShell
pytest tests/test_api.py
🔐 Sample Credentials
Username: admin
Password: password123
