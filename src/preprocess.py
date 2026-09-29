"""
preprocess.py
Module 1: Data Preprocessing & Cleaning pipeline.
Handles missing values, duplicate removal, date conversions, feature engineering,
categorical encoding, and numerical scaling.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_raw_data(data_dir="data"):
    bookings_path = os.path.join(data_dir, "raw", "sound_studio_bookings.csv")
    customers_path = os.path.join(data_dir, "raw", "sound_studio_customers.csv")
    
    if not os.path.exists(bookings_path) or not os.path.exists(customers_path):
        raise FileNotFoundError("Raw data files missing. Please run src/data_generator.py first.")
        
    df_bookings = pd.read_csv(bookings_path)
    df_customers = pd.read_csv(customers_path)
    return df_bookings, df_customers

def clean_bookings(df_bookings):
    """Clean and preprocess booking transactions."""
    df = df_bookings.copy()
    
    # 1. Deduplication
    initial_len = len(df)
    df = df.drop_duplicates(subset=["booking_id"])
    
    # 2. Date parsing
    df["booking_date"] = pd.to_datetime(df["booking_date"])
    df["year"] = df["booking_date"].dt.year
    df["month"] = df["booking_date"].dt.month
    df["month_name"] = df["booking_date"].dt.month_name()
    df["day_of_week"] = df["booking_date"].dt.day_name()
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    
    # 3. Handle any potential anomalies/nulls
    df["amount_paid"] = pd.to_numeric(df["amount_paid"], errors="coerce")
    df["duration_hours"] = pd.to_numeric(df["duration_hours"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    
    df["amount_paid"] = df["amount_paid"].fillna(df["amount_paid"].median())
    df["duration_hours"] = df["duration_hours"].fillna(df["duration_hours"].median())
    df["rating"] = df["rating"].fillna(df["rating"].mean())
    
    # 4. Feature engineering: Spend per hour
    df["spend_per_hour"] = np.where(df["duration_hours"] > 0, df["amount_paid"] / df["duration_hours"], df["amount_paid"])
    
    return df

def clean_customers(df_customers):
    """Clean and preprocess customer profiles for ML & segmentation."""
    df = df_customers.copy()
    
    # 1. Deduplication
    df = df.drop_duplicates(subset=["customer_id"])
    
    # 2. Impute any missing values
    df["age"] = df["age"].fillna(df["age"].median())
    df["gender"] = df["gender"].fillna("Other")
    df["city"] = df["city"].fillna("Chennai")
    df["client_type"] = df["client_type"].fillna("Independent Musician")
    df["primary_service"] = df["primary_service"].fillna("Recording")
    df["avg_rating"] = df["avg_rating"].fillna(df["avg_rating"].mean())
    df["total_spend"] = df["total_spend"].fillna(0)
    df["days_since_last_visit"] = df["days_since_last_visit"].fillna(180)
    
    # 3. Age Binning for behavioral analysis
    age_bins = [0, 24, 34, 49, 100]
    age_labels = ["18-24 (Gen Z)", "25-34 (Young Pro)", "35-49 (Mid Career)", "50+ (Veteran)"]
    df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels)
    
    # 4. Spend category
    spend_quantiles = df["total_spend"].quantile([0.33, 0.66]).values
    def categorize_spend(val):
        if val <= spend_quantiles[0]:
            return "Low Spend"
        elif val <= spend_quantiles[1]:
            return "Medium Spend"
        else:
            return "High Spend"
    df["spend_tier"] = df["total_spend"].apply(categorize_spend)
    
    return df

def preprocess_and_encode(df_customers, models_dir="models", is_train=True):
    """
    Encode categorical features and fit/apply StandardScaler.
    Saves scaler and encoders for production inference.
    """
    df = df_customers.copy()
    os.makedirs(models_dir, exist_ok=True)
    
    cat_cols = ["gender", "city", "client_type", "primary_service"]
    encoders = {}
    
    if is_train:
        for col in cat_cols:
            le = LabelEncoder()
            df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        joblib.dump(encoders, os.path.join(models_dir, "label_encoders.pkl"))
    else:
        encoders_path = os.path.join(models_dir, "label_encoders.pkl")
        if os.path.exists(encoders_path):
            encoders = joblib.load(encoders_path)
            for col in cat_cols:
                le = encoders.get(col)
                if le:
                    # handle unseen categories
                    df[f"{col}_encoded"] = df[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
                else:
                    df[f"{col}_encoded"] = 0
                    
    feature_cols = [
        "age", "days_since_last_visit", "visit_count", "total_spend",
        "avg_booking_spend", "total_hours", "avg_duration_hours",
        "avg_rating", "avg_discount_received",
        "gender_encoded", "city_encoded", "client_type_encoded", "primary_service_encoded"
    ]
    
    if is_train:
        scaler = StandardScaler()
        df_scaled_features = scaler.fit_transform(df[feature_cols])
        joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
        joblib.dump(feature_cols, os.path.join(models_dir, "feature_columns.pkl"))
    else:
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            df_scaled_features = scaler.transform(df[feature_cols])
        else:
            df_scaled_features = None
            
    return df, df_scaled_features, feature_cols

def run_preprocessing_pipeline(data_dir="data", models_dir="models"):
    print("-> Loading raw datasets...")
    df_bookings, df_customers = load_raw_data(data_dir)
    
    print("-> Cleaning Bookings...")
    cleaned_bookings = clean_bookings(df_bookings)
    
    print("-> Cleaning Customers...")
    cleaned_customers = clean_customers(df_customers)
    
    print("-> Encoding features & scaling...")
    processed_customers, _, _ = preprocess_and_encode(cleaned_customers, models_dir=models_dir, is_train=True)
    
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    cleaned_bookings.to_csv(os.path.join(processed_dir, "cleaned_bookings.csv"), index=False)
    processed_customers.to_csv(os.path.join(processed_dir, "cleaned_customers.csv"), index=False)
    
    print("Preprocessing completed successfully!")
    print(f"Cleaned bookings saved: {len(cleaned_bookings)} rows")
    print(f"Cleaned customers saved: {len(processed_customers)} rows")
    return cleaned_bookings, processed_customers

if __name__ == "__main__":
    run_preprocessing_pipeline()
