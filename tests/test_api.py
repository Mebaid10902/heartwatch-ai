import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Sample credentials (you must have a user with this in your DB or mock the login)
USERNAME = "admin"
PASSWORD = "password123"

@pytest.fixture
def token():
    response = client.post("/login", data={"username": USERNAME, "password": PASSWORD})
    assert response.status_code == 200
    return response.json()["access_token"]

def test_login():
    response = client.post("/login", data={"username": USERNAME, "password": PASSWORD})
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_predict(token):
    headers = {"Authorization": f"Bearer {token}"}
    sample_input = {
        "age": 60,
        "sex": 1,
        "cp": 3,
        "trestbps": 140,
        "chol": 250,
        "fbs": 1,
        "restecg": 0,
        "thalch": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 1,
        "ca": 0,
        "thal": 2
    }
    response = client.post("/predict", json=sample_input, headers=headers)
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_feedback(token):
    headers = {"Authorization": f"Bearer {token}"}
    sample_feedback = {
        "age": 60,
        "sex": 1,
        "cp": 3,
        "trestbps": 140,
        "chol": 250,
        "fbs": 1,
        "restecg": 0,
        "thalch": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 1,
        "ca": 0,
        "thal": 2,
        "prediction": 1
    }
    response = client.post("/feedback", json=sample_feedback, headers=headers)
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_retrain(token):
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/retrain", headers=headers)
    assert response.status_code == 200
    assert response.json()["message"].startswith("Model retrained")
