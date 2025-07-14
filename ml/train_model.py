import os, joblib, warnings, random
from collections import Counter

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from imblearn.over_sampling import RandomOverSampler

from app.database import SessionLocal
from app.models import Feedback

warnings.filterwarnings("ignore")
random.seed(44)

# ------------------------------------------------------------------ #
# 1. LOAD CSV + SQLITE FEEDBACK AND CLEANING
# ------------------------------------------------------------------ #
# ------------------------------------------------------------------ #
# 1. LOAD CSV  +  SQLITE FEEDBACK  AND  CLEANING
# ------------------------------------------------------------------ #
def load_and_prepare_data() -> tuple:
    """
    Returns:
        (X_train, X_test, y_train, y_test), FEATURES
    """
    # ---------- 1.1 read primary CSV --------------------------------
    df = pd.read_csv("data/heart.csv")

    # drop ID if present
    if "id" in df.columns:
        df.drop(columns="id", inplace=True)
    if "dataset" in df.columns:
        df.drop(columns="dataset", inplace=True)

    # ---------- 1.2 pull feedback rows from SQLite ------------------
    db = SessionLocal()
    feedback_rows = db.query(Feedback).all()
    db.close()

    if feedback_rows and len(feedback_rows) >= 50:          # only if we have enough rows
        fb_df = pd.DataFrame(
            [{
                "age": fb.age,  "sex": fb.sex,
                "cp": fb.cp,    "trestbps": fb.trestbps,  "chol": fb.chol,
                "fbs": fb.fbs,  "restecg": fb.restecg,    "thalch": fb.thalch,
                "exang": fb.exang,  "oldpeak": fb.oldpeak,
                "slope": fb.slope,  "ca": fb.ca,  "thal": fb.thal,
                "target": fb.prediction
            } for fb in feedback_rows]
        ).drop_duplicates()

        df = pd.concat([df, fb_df], ignore_index=True)
        print(f"✅ Integrated {len(fb_df)} feedback rows")
    else:
        print(f"ℹ️ Skipping feedback integration (only {len(feedback_rows)} found)")

    # ---------- 1.3 tidy / rename columns ---------------------------
    if "num" in df.columns and "target" not in df.columns:
        df.rename(columns={"num": "target"}, inplace=True)

    # ---------- 1.4 categorical mappings ----------------------------
    # maps *after* we coerce to str.lower()
    MAP_SEX       = {"male": 0, "female": 1}
    MAP_CP        = {"typical angina": 0, "atypical angina": 1,
                     "non-anginal": 2,"non-anginal pain": 2,
                     "asymptomatic": 3}
    MAP_RESTECG   = {"normal": 0,"st-t wave abnormality": 1,
                    "lv hypertrophy": 2,
                     "left ventricular hypertrophy": 2,
                     }
    MAP_SLOPE     = {"upsloping": 0, "flat": 1, "downsloping": 2}
    MAP_THAL      = {"normal": 0, "fixed defect": 1,
                     "reversable defect": 2, "reversible defect": 2}

    # harmonise boolean‑like strings
    df["fbs"]  = df["fbs"].astype(str).str.lower().map({"true": 1, "false": 0}).fillna(0).astype(int)
    df["exang"]= df["exang"].astype(str).str.lower().map({"true": 1, "false": 0}).fillna(0).astype(int)

    cat_mappings = {
        "sex": MAP_SEX,
        "cp": MAP_CP,
        "restecg": MAP_RESTECG,
        "slope": MAP_SLOPE,
        "thal": MAP_THAL,
    }

    for col, mapper in cat_mappings.items():
        df[col] = (
            df[col]
            .astype(str)                # ensure string
            .str.strip()                # remove spaces
            .str.lower()                # lower‑case
            .map(mapper)                # apply mapping
        )
        # fill unmapped (NaN) with column median
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val).astype(int)
    # ---------- 1.5 numeric columns / label -------------------------
    FEATURES = ["age","sex","cp","trestbps","chol","fbs","restecg",
                "thalch","exang","oldpeak","slope","ca","thal"]

    df.dropna(subset=["target"], inplace=True)
    df["target"] = (df["target"] > 0).astype(int)

    # optional 2:1 majority cap
    vc = df["target"].value_counts()
    if len(vc) == 2 and vc.max() / vc.min() > 2:
        maj_label = vc.idxmax()
        df_maj = df[df["target"] == maj_label].sample(int(2 * vc.min()), random_state=44)
        df_min = df[df["target"] != maj_label]
        df = pd.concat([df_maj, df_min])

    X = df[FEATURES].fillna(df[FEATURES].median(numeric_only=True))
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=44
    )
    print("✅ Class distribution train:", Counter(y_train))

    return X_train, X_test, y_train, y_test, FEATURES


# ------------------------------------------------------------------ #
# 2. TRAIN AND LOG MODELS
# ------------------------------------------------------------------ #
def train_and_log_models(X_train, X_test, y_train, y_test, feature_cols):
    ros = RandomOverSampler(random_state=44)
    X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)

    mlflow.set_experiment("heart_disease_prediction")
    best, best_name, best_auc, best_acc = None, "", 0, 0

    CANDIDATES = {
        "RandomForest": RandomForestClassifier(n_estimators=400, max_depth=None, random_state=44),
        "LogisticRegression": LogisticRegression(max_iter=500, class_weight="balanced"),
    }

    with mlflow.start_run(run_name="RF_LR_Baselines"):
        for name, model in CANDIDATES.items():
            pipe = Pipeline([("scale", StandardScaler()), ("clf", model)])
            pipe.fit(X_train_bal, y_train_bal)
            auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
            acc = accuracy_score(y_test, pipe.predict(X_test))
            f1 = f1_score(y_test, pipe.predict(X_test))

            mlflow.log_metric(f"{name}_auc", auc)
            mlflow.log_metric(f"{name}_acc", acc)
            mlflow.log_metric(f"{name}_f1", f1)

            if auc > best_auc and acc > best_acc:
                best, best_name, best_auc, best_acc = pipe, name, auc, acc
            print(f"✔ {name}: AUC={auc:.3f}  ACC={round(acc * 100, 2)}%")

    xgb_base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=500,
        random_state=44
    )
    param_dist = {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.02, 0.05, 0.1],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "gamma": [0, 1, 5]
    }

    min_class_count = min(Counter(y_train_bal).values())
    cv_folds = min(5, min_class_count) if min_class_count >= 2 else 2

    tuner = RandomizedSearchCV(
        xgb_base,
        param_dist,
        n_iter=20,
        cv=cv_folds,
        scoring="roc_auc",
        verbose=0,
        random_state=44,
        n_jobs=-1
    )

    tuner.fit(X_train_bal, y_train_bal)

    xgb_best = tuner.best_estimator_
    auc = roc_auc_score(y_test, xgb_best.predict_proba(X_test)[:, 1])
    acc = accuracy_score(y_test, xgb_best.predict(X_test))
    f1 = f1_score(y_test, xgb_best.predict(X_test))

    with mlflow.start_run(run_name="XGB_Tuned"):
        mlflow.log_params(tuner.best_params_)
        mlflow.log_metrics({"auc": auc, "accuracy": acc, "f1": f1})

    if auc > best_auc and acc > best_acc:
        best, best_name, best_auc, best_acc = xgb_best, "XGB_Tuned", auc, acc

    print(confusion_matrix(y_test, best.predict(X_test)))
    print(f"🏆 Best model: {best_name} (AUC={best_auc:.3f}) ACC={round(acc * 100, 2)}%")

    os.makedirs("models", exist_ok=True)
    joblib.dump(best, "models/model.pkl")
    print("✅ model.pkl updated")

    with mlflow.start_run(run_name=f"Register_{best_name}"):
        mlflow.sklearn.log_model(best, "model", registered_model_name="heart_model")
        mlflow.log_metric("auc", best_auc)

# ------------------------------------------------------------------ #
# 3. MAIN
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, FEATS = load_and_prepare_data()
    train_and_log_models(X_train, X_test, y_train, y_test, FEATS)
