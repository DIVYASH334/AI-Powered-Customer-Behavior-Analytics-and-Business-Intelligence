"""
segmentation.py
Module 3: Customer Segmentation using RFM (Recency, Frequency, Monetary)
and Unsupervised Machine Learning (K-Means Clustering).
Identifies:
  - High-Value Champions (High frequency, high monetary, low recency)
  - Loyal Regulars (Consistent bookings, good spend)
  - Potential / Promising (Recent first-timers, good spend potential)
  - At-Risk / Hibernating (High recency, low recent visits)
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def calculate_rfm_metrics(df_customers):
    """
    Extract RFM metrics:
    - Recency: days_since_last_visit
    - Frequency: visit_count
    - Monetary: total_spend
    """
    df = df_customers.copy()
    
    # Ensure columns exist
    required_cols = ["customer_id", "days_since_last_visit", "visit_count", "total_spend"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")
            
    # Calculate RFM quantiles / scores (1 to 4)
    # Recency: lower is better (rank reverse)
    df["R_Score"] = pd.qcut(df["days_since_last_visit"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
    # Frequency: higher is better
    df["F_Score"] = pd.qcut(df["visit_count"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    # Monetary: higher is better
    df["M_Score"] = pd.qcut(df["total_spend"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    
    df["RFM_Score"] = df["R_Score"].astype(str) + df["F_Score"].astype(str) + df["M_Score"].astype(str)
    df["RFM_Sum"] = df["R_Score"] + df["F_Score"] + df["M_Score"]
    
    return df

def train_kmeans_segmentation(df_rfm, n_clusters=4, models_dir="models"):
    """
    Train K-Means clustering on standardized RFM features + PCA for 2D/3D visualization.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    features = ["days_since_last_visit", "visit_count", "total_spend"]
    X = df_rfm[features].copy()
    
    # Standardize
    rfm_scaler = StandardScaler()
    X_scaled = rfm_scaler.fit_transform(X)
    
    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df_rfm["cluster"] = clusters
    
    # Compute PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    df_rfm["pca_x"] = pca_coords[:, 0]
    df_rfm["pca_y"] = pca_coords[:, 1]
    
    # Profile clusters to assign meaningful business segment labels
    cluster_means = df_rfm.groupby("cluster")[features].mean()
    
    # Mapping based on monetary and recency
    # Order clusters by monetary value and recency
    rankings = cluster_means.sort_values(by="total_spend", ascending=False).index.tolist()
    
    # 0 = Top Spender/Loyal, etc.
    segment_map = {}
    segment_map[rankings[0]] = "High-Value Champions"
    segment_map[rankings[1]] = "Loyal Regulars"
    
    # Differentiate between At-Risk and Promising for remaining two
    remaining = [rankings[2], rankings[3]]
    if cluster_means.loc[remaining[0], "days_since_last_visit"] > cluster_means.loc[remaining[1], "days_since_last_visit"]:
        segment_map[remaining[0]] = "At-Risk / Inactive"
        segment_map[remaining[1]] = "Potential / Promising"
    else:
        segment_map[remaining[0]] = "Potential / Promising"
        segment_map[remaining[1]] = "At-Risk / Inactive"
        
    df_rfm["segment_name"] = df_rfm["cluster"].map(segment_map)
    
    # Assign Actionable Strategy recommendations
    strategy_map = {
        "High-Value Champions": "VIP Studio Perks, Priority Room Booking, Personal Sound Engineer, Exclusive Mastering Discounts",
        "Loyal Regulars": "Loyalty Points, Album Bundle Discounts, Referral Bonus, Free Acoustic Consultations",
        "Potential / Promising": "Follow-up Feedback Call, 15% Welcome-Back Voucher on Next Recording, Social Community Invites",
        "At-Risk / Inactive": "Win-back Re-engagement Email, Free 1-hour Rehearsal Voucher, Inactive Customer Feedback Survey"
    }
    df_rfm["actionable_strategy"] = df_rfm["segment_name"].map(strategy_map)
    
    # Save artifacts
    joblib.dump(kmeans, os.path.join(models_dir, "kmeans_model.pkl"))
    joblib.dump(rfm_scaler, os.path.join(models_dir, "rfm_scaler.pkl"))
    joblib.dump(pca, os.path.join(models_dir, "rfm_pca.pkl"))
    joblib.dump(segment_map, os.path.join(models_dir, "segment_mapping.pkl"))
    
    return df_rfm, kmeans, cluster_means

def run_segmentation_pipeline(data_dir="data", models_dir="models"):
    cleaned_customers_path = os.path.join(data_dir, "processed", "cleaned_customers.csv")
    if not os.path.exists(cleaned_customers_path):
        raise FileNotFoundError("Cleaned customers file not found. Run preprocessing first.")
        
    df_customers = pd.read_csv(cleaned_customers_path)
    
    print("-> Calculating RFM Scores...")
    df_rfm = calculate_rfm_metrics(df_customers)
    
    print("-> Fitting K-Means Clustering (k=4)...")
    segmented_df, kmeans, cluster_summary = train_kmeans_segmentation(df_rfm, n_clusters=4, models_dir=models_dir)
    
    out_file = os.path.join(data_dir, "processed", "rfm_segmented_customers.csv")
    segmented_df.to_csv(out_file, index=False)
    
    print(f"Segmentation complete! Saved to {out_file}")
    print("\nSegment Profiles:")
    print(segmented_df["segment_name"].value_counts())
    return segmented_df

if __name__ == "__main__":
    run_segmentation_pipeline()
