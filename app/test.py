import joblib
import numpy as np

# Load your trained model
model = joblib.load("models/model.pkl")

# Sample JSONs that likely predict "No Heart Disease"
inputs = [
    {
        "age": 45, "sex": 0, "cp": 1, "trestbps": 120, "chol": 200,
        "fbs": 0, "restecg": 1, "thalch": 170, "exang": 0,
        "oldpeak": 0.1, "slope": 1, "ca": 0, "thal": 1
    },
    {
        "age": 39, "sex": 1, "cp": 2, "trestbps": 110, "chol": 180,
        "fbs": 0, "restecg": 0, "thalch": 190, "exang": 0,
        "oldpeak": 0.0, "slope": 0, "ca": 0, "thal": 1
    }
]

# Prepare data for prediction
X = np.array([[i[k] for k in i] for i in inputs])
preds = model.predict(X)

# Display predictions
for i, p in enumerate(preds):
    print(f"Input {i+1} Prediction:", "Heart Disease" if p == 1 else "No Heart Disease")
