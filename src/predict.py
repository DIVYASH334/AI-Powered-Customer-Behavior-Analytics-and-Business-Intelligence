"""
predict.py
Module 4: Predictive Modeling for Customer Retention & Churn.
Trains, evaluates, and compares multiple Machine Learning models:
  - Logistic Regression (Baseline linear classifier)
  - Random Forest Classifier (Ensemble tree-based)
  - Gradient Boosting Classifier (Boosting ensemble)
Uses genuine behavioral features (no target leakage from visit_count or total_spend)
to predict retention probability when evaluating a customer.
Computes comprehensive metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix.
Saves serialized models and metrics JSON for dashboard visualization and live predictions.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

# Features selected without target leakage (no visit_count, no total_spend)
FEATURE_COLUMNS = [
    "age",
    "days_since_last_visit",
    "avg_booking_spend",
    "avg_duration_hours",
    "avg_rating",
    "avg_discount_received",
    "gender_encoded",
    "city_encoded",
    "client_type_encoded",
    "primary_service_encoded"
]

def load_processed_data(data_path="data/processed/cleaned_customers.csv"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {data_path}")
    return pd.read_csv(data_path)

def evaluate_model(model, X_test, y_test, model_name):
    """Calculate all standard evaluation metrics for binary classification."""
    y_pred = model.predict(X_test)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, y_prob))
    else:
        y_prob = y_pred
        auc = 0.5
        
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    metrics = {
        "model_name": model_name,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm
    }
    return metrics

def train_retention_models(df, models_dir="models"):
    """
    Train models to predict `repeat_customer` (1 = Repeat, 0 = Single visit).
    """
    os.makedirs(models_dir, exist_ok=True)
    
    X = df[FEATURE_COLUMNS].copy()
    y = df["repeat_customer"].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Scale features for linear model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_metrics = evaluate_model(lr, X_test_scaled, y_test, "Logistic Regression")
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_split=4, random_state=42)
    rf.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf, X_test, y_test, "Random Forest")
    
    # 3. Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    gb_metrics = evaluate_model(gb, X_test, y_test, "Gradient Boosting")
    
    # Feature importances from Random Forest
    feat_importances = dict(zip(FEATURE_COLUMNS, [round(float(x), 4) for x in rf.feature_importances_]))
    sorted_importances = dict(sorted(feat_importances.items(), key=lambda item: item[1], reverse=True))
    
    # Save models & feature list
    joblib.dump(lr, os.path.join(models_dir, "repeat_classifier_lr.pkl"))
    joblib.dump(rf, os.path.join(models_dir, "repeat_classifier_rf.pkl"))
    joblib.dump(gb, os.path.join(models_dir, "repeat_classifier_gb.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "model_feature_scaler.pkl"))
    joblib.dump(FEATURE_COLUMNS, os.path.join(models_dir, "retention_features.pkl"))
    
    comparison = [lr_metrics, rf_metrics, gb_metrics]
    
    return {
        "models": {"lr": lr, "rf": rf, "gb": gb},
        "metrics": comparison,
        "feature_importances": sorted_importances
    }

def train_churn_model(df, models_dir="models"):
    """
    Train a secondary classifier to predict `churn_risk` (1 = At-Risk, 0 = Retained).
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df["churn_risk"].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    churn_rf = RandomForestClassifier(n_estimators=120, max_depth=5, random_state=42)
    churn_rf.fit(X_train, y_train)
    churn_metrics = evaluate_model(churn_rf, X_test, y_test, "Random Forest (Churn)")
    
    joblib.dump(churn_rf, os.path.join(models_dir, "churn_classifier_rf.pkl"))
    return churn_rf, churn_metrics

def run_prediction_pipeline(data_path="data/processed/cleaned_customers.csv", models_dir="models"):
    print("-> Loading preprocessed data for modeling...")
    df = load_processed_data(data_path)
    
    print("-> Training Customer Retention / Repeat Prediction Models...")
    retention_results = train_retention_models(df, models_dir=models_dir)
    
    print("-> Training Customer Churn Risk Model...")
    _, churn_metrics = train_churn_model(df, models_dir=models_dir)
    
    report = {
        "retention_models": retention_results["metrics"],
        "churn_model": churn_metrics,
        "feature_importances": retention_results["feature_importances"],
        "feature_names": FEATURE_COLUMNS
    }
    
    metrics_path = os.path.join(models_dir, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Models and metrics successfully saved to {models_dir}/")
    print("\n=== Model Comparison Results ===")
    for m in retention_results["metrics"]:
        print(f"[{m['model_name']}] Accuracy: {m['accuracy']*100:.2f}% | F1: {m['f1_score']:.4f} | ROC-AUC: {m['roc_auc']:.4f}")
        
    print(f"\n[Churn Model] Accuracy: {churn_metrics['accuracy']*100:.2f}% | F1: {churn_metrics['f1_score']:.4f} | ROC-AUC: {churn_metrics['roc_auc']:.4f}")
    return report

def predict_single_customer(customer_input, models_dir="models"):
    """
    Inference helper for live prediction in the Streamlit app.
    customer_input: dict with customer attributes
    """
    rf_retention = joblib.load(os.path.join(models_dir, "repeat_classifier_rf.pkl"))
    rf_churn = joblib.load(os.path.join(models_dir, "churn_classifier_rf.pkl"))
    encoders = joblib.load(os.path.join(models_dir, "label_encoders.pkl"))
    
    # Safe categorical encoding
    def safe_encode(col_name, val, default_val):
        if col_name in encoders and val in encoders[col_name].classes_:
            return int(encoders[col_name].transform([val])[0])
        elif col_name in encoders and default_val in encoders[col_name].classes_:
            return int(encoders[col_name].transform([default_val])[0])
        return 0
        
    gender_enc = safe_encode("gender", customer_input.get("gender"), "Male")
    city_enc = safe_encode("city", customer_input.get("city"), "Chennai")
    client_enc = safe_encode("client_type", customer_input.get("client_type"), "Independent Musician")
    service_enc = safe_encode("primary_service", customer_input.get("primary_service"), "Recording")
    
    features = pd.DataFrame([{
        "age": customer_input.get("age", 28),
        "days_since_last_visit": customer_input.get("days_since_last_visit", 30),
        "avg_booking_spend": customer_input.get("avg_booking_spend", 4500),
        "avg_duration_hours": customer_input.get("avg_duration_hours", 3),
        "avg_rating": customer_input.get("avg_rating", 4.5),
        "avg_discount_received": customer_input.get("avg_discount_received", 5.0),
        "gender_encoded": gender_enc,
        "city_encoded": city_enc,
        "client_type_encoded": client_enc,
        "primary_service_encoded": service_enc
    }])[FEATURE_COLUMNS]
    
    repeat_prob = float(rf_retention.predict_proba(features)[0][1])
    repeat_pred = int(repeat_prob >= 0.5)
    
    churn_prob = float(rf_churn.predict_proba(features)[0][1])
    churn_pred = int(churn_prob >= 0.5)
    
    return {
        "repeat_prediction": "Yes (Likely to Return)" if repeat_pred == 1 else "No (Single Visit / Churn Tendency)",
        "repeat_probability": round(repeat_prob * 100, 1),
        "churn_risk": "High Risk" if churn_pred == 1 else "Low Risk / Engaged",
        "churn_probability": round(churn_prob * 100, 1)
    }

if __name__ == "__main__":
    run_prediction_pipeline()
