import streamlit as st
import os
import requests

# --- CONFIG ---
API_URL = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# --- Session State ---
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None
if "page" not in st.session_state:
    st.session_state.page = "login"

# --- PAGE NAVIGATION ---
if st.session_state.token:
    page = st.sidebar.radio("📂 Pages", ["🏠 Predict", "📝 Submit Feedback","🔁 Retrain Model", "🚪 Logout"])
    if page == "🏠 Predict":
        st.session_state.page = "predict"
    elif page == "📝 Submit Feedback":
        st.session_state.page = "feedback"
    elif page == "🔁 Retrain Model":
        st.session_state.page = "retrain"
    elif page == "🚪 Logout":
        st.session_state.token = None
        st.session_state.username = None
        st.session_state.page = "login"
        st.rerun()

# --- LOGIN PAGE ---
if st.session_state.page == "login":
    st.title("🔐 Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_btn = st.form_submit_button("Login")

    if login_btn:
        res = requests.post(f"{API_URL}/login", data={"username": username, "password": password})
        if res.status_code == 200:
            st.session_state.token = res.json()["access_token"]
            st.session_state.username = username
            st.success("✅ Logged in successfully!")
            st.session_state.page = "predict"
            st.rerun()
        else:
            st.error("❌ Invalid credentials")
    st.stop()

# --- COMMON ENCODER ---
def encode_input(form):
    data = {
        "age": form["age"],
        "sex": 1 if form["sex"] == "Male" else 0,
        "cp": {"typical angina": 0, "atypical angina": 1, "non-anginal pain": 2, "asymptomatic": 3}[form["cp"]],
        "trestbps": form["trestbps"],
        "chol": form["chol"],
        "fbs": int(form["fbs"]),
        "restecg": {"normal": 0, "ST-T wave abnormality": 1, "left ventricular hypertrophy": 2}[form["restecg"]],
        "thalch": form["thalch"],
        "exang": int(form["exang"]),
        "oldpeak": form["oldpeak"],
        "slope": {"upsloping": 0, "flat": 1, "downsloping": 2}[form["slope"]],
        "ca": form["ca"],
        "thal": {"normal": 1, "fixed defect": 2, "reversible defect": 3}[form["thal"]],
    }
    # Add prediction only if present (feedback page)
    if "prediction" in form:
        data["prediction"] = form["prediction"]

    return data

# --- PREDICT PAGE ---
if st.session_state.page == "predict":
    st.title("💓 Heart Disease Prediction")

    with st.form("prediction_form"):
        st.subheader("📋 Enter Patient Data")
        form = {
            "age": st.slider("Age", 20, 90, 50),
            "sex": st.selectbox("Sex", ["Male", "Female"]),
            "cp": st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"]),
            "trestbps": st.number_input("Resting BP", 80, 200, 120),
            "chol": st.number_input("Cholesterol", 100, 400, 200),
            "fbs": st.checkbox("Fasting Blood Sugar > 120 mg/dl"),
            "restecg": st.selectbox("Rest ECG", ["normal", "ST-T wave abnormality", "left ventricular hypertrophy"]),
            "thalch": st.number_input("Max Heart Rate", 70, 210, 150),
            "exang": st.checkbox("Exercise Induced Angina"),
            "oldpeak": st.number_input("Oldpeak", 0.0, 6.0, 1.0, step=0.1),
            "slope": st.selectbox("Slope", ["upsloping", "flat", "downsloping"]),
            "ca": st.selectbox("Major Vessels (ca)", [0, 1, 2, 3]),
            "thal": st.selectbox("Thalassemia", ["normal", "fixed defect", "reversible defect"])
        }
        submit = st.form_submit_button("🔮 Predict")

    if submit:
        data = encode_input(form)
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        try:
            # 🎯 Step 1: Predict
            res = requests.post(f"{API_URL}/predict", json=data, headers=headers)
            if res.status_code == 200:
                result = res.json()
                predicted_class = result["predicted_class"]
                st.success(f"🎯 Prediction: {result['prediction']}")
                st.info(f"🔢 Predicted class: {predicted_class}")

                # 💡 Step 2: Get LLM Advice
                advice_payload = {
                    "prediction": str(predicted_class),
                    "patient_data": data,
                }
                advice_res = requests.post(f"{API_URL}/advise", json=advice_payload, headers=headers)
                if advice_res.status_code == 200:
                    advice = advice_res.json()["advice"]
                    st.markdown("🧠 **LLM Medical Advice:**")
                    st.success(advice)
                else:
                    st.warning("⚠️ Advice could not be retrieved.")
            else:
                st.error(f"❌ Prediction API Error: {res.status_code}")
        except Exception as e:
            st.error(f"🔌 Connection Error: {e}")

# --- FEEDBACK PAGE ---
elif st.session_state.page == "feedback":
    st.title("📝 Submit Feedback")

    with st.form("feedback_form"):
        st.subheader("📋 Patient Data + Your Judgment")
        form = {
            "age": st.slider("Age", 20, 90, 50),
            "sex": st.selectbox("Sex", ["Male", "Female"], ),
            "cp": st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"] ),
            "trestbps": st.number_input("Resting BP", 80, 200, 120),
            "chol": st.number_input("Cholesterol", 100, 400, 200 ),
            "fbs": st.checkbox("Fasting Blood Sugar > 120 mg/dl"),
            "restecg": st.selectbox("Rest ECG", ["normal", "ST-T wave abnormality", "left ventricular hypertrophy"]),
            "thalch": st.number_input("Max Heart Rate", 70, 210, 150),
            "exang": st.checkbox("Exercise Induced Angina" ),
            "oldpeak": st.number_input("Oldpeak", 0.0, 6.0, 1.0),
            "slope": st.selectbox("Slope", ["upsloping", "flat", "downsloping"]),
            "ca": st.selectbox("Major Vessels (ca)", [0, 1, 2, 3]),
            "thal": st.selectbox("Thalassemia", ["normal", "fixed defect", "reversible defect"]),
            "prediction": st.selectbox("Your Manual prediction", [0, 1])
        }
        send = st.form_submit_button("💾 Submit Feedback")

        if send:
            data = encode_input(form)
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            try:
                res = requests.post(f"{API_URL}/feedback", json=data, headers=headers)
                if res.status_code == 200:
                    result = res.json()
                    st.success(f"✅ Feedback stored!")
                else:
                    st.error(f"❌ API Error: {res.status_code}")
            except Exception as e:
                st.error(f"🔌 Connection Error: {e}")
elif st.session_state.page == "retrain":
    st.title("🔁 Retrain Model")
    with st.form("retrain_form"):
        send = st.form_submit_button("🔁 Retrain Model")
        if send :
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            try:
                res = requests.post(f"{API_URL}/retrain", headers=headers)
                if res.status_code == 200:
                    result = res.json()
                    st.success(f"✅ Model retrained successfully!")
                else:
                    st.error(f"❌ Retrain failed: {res.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"🔌 Connection error during retraining: {e}")    
