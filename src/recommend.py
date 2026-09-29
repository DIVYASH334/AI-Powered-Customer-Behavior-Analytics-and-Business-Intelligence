"""
recommend.py
Module 5: Studio Service Recommendation System.
Combines:
  1. Item-Based Collaborative Filtering (Cosine Similarity on User-Service interaction matrix)
  2. Service Co-occurrence / Market Basket Affinity ("Clients who booked X also booked Y")
  3. Client Persona & Segment-based Heuristics
Produces personalized, ranked service recommendations with transparent business explanations.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ALL_SERVICES = [
    "Recording",
    "Mixing",
    "Mastering",
    "Rehearsal Space",
    "Voiceover / Dubbing",
    "Music Production",
    "Podcast Recording"
]

SERVICE_DESCRIPTIONS = {
    "Recording": "High-end multi-track tracking with Neumann, AKG mics & SSL preamps.",
    "Mixing": "Precision balance, analog summing, surgical EQ and dynamic spatialization.",
    "Mastering": "Loudness optimization, Apple Digital Masters / Spotify compliance.",
    "Rehearsal Space": "Acoustically treated live room with drum kit, amps & PA system.",
    "Voiceover / Dubbing": "Dead-quiet vocal booth with Source-Connect & ADR capabilities.",
    "Music Production": "Full arrangement, beat crafting, session musician coordination.",
    "Podcast Recording": "Multi-mic Shure SM7B setup with video-sync recording suite."
}

def build_service_interaction_matrix(bookings_path="data/processed/cleaned_bookings.csv"):
    if not os.path.exists(bookings_path):
        raise FileNotFoundError(f"Bookings data not found at {bookings_path}")
        
    df_bookings = pd.read_csv(bookings_path)
    
    # Interaction matrix: Rows = Customers, Columns = Service Types, Values = Booking Frequency or Spend
    interaction_df = df_bookings.pivot_table(
        index="customer_id",
        columns="service_type",
        values="booking_id",
        aggfunc="count",
        fill_value=0
    )
    
    # Ensure all services exist in columns
    for s in ALL_SERVICES:
        if s not in interaction_df.columns:
            interaction_df[s] = 0
            
    # Compute Item-Item Cosine Similarity matrix (Service x Service)
    service_matrix = interaction_df.T  # Services as rows, Customers as columns
    similarity = cosine_similarity(service_matrix)
    similarity_df = pd.DataFrame(similarity, index=service_matrix.index, columns=service_matrix.index)
    
    # Co-occurrence probability matrix
    co_occur_matrix = pd.DataFrame(0.0, index=ALL_SERVICES, columns=ALL_SERVICES)
    customers_grouped = df_bookings.groupby("customer_id")["service_type"].unique()
    
    for services_used in customers_grouped:
        for s1 in services_used:
            for s2 in services_used:
                if s1 != s2:
                    co_occur_matrix.loc[s1, s2] += 1
                    
    # Normalize co-occurrence
    for s1 in ALL_SERVICES:
        row_sum = co_occur_matrix.loc[s1].sum()
        if row_sum > 0:
            co_occur_matrix.loc[s1] = co_occur_matrix.loc[s1] / row_sum
            
    return interaction_df, similarity_df, co_occur_matrix

def train_and_save_recommender(data_dir="data", models_dir="models"):
    os.makedirs(models_dir, exist_ok=True)
    bookings_path = os.path.join(data_dir, "processed", "cleaned_bookings.csv")
    
    interaction_df, similarity_df, co_occur_matrix = build_service_interaction_matrix(bookings_path)
    
    joblib.dump(interaction_df, os.path.join(models_dir, "recommender_interaction.pkl"))
    joblib.dump(similarity_df, os.path.join(models_dir, "service_similarity.pkl"))
    joblib.dump(co_occur_matrix, os.path.join(models_dir, "service_co_occurrence.pkl"))
    
    print("Recommender system artifacts successfully saved!")
    return similarity_df, co_occur_matrix

def get_recommendations_for_customer(customer_id, top_n=3, models_dir="models", data_dir="data"):
    """
    Generate personalized recommendations for an existing registered customer.
    """
    interaction_path = os.path.join(models_dir, "recommender_interaction.pkl")
    sim_path = os.path.join(models_dir, "service_similarity.pkl")
    cust_path = os.path.join(data_dir, "processed", "rfm_segmented_customers.csv")
    
    if not os.path.exists(sim_path) or not os.path.exists(interaction_path):
        train_and_save_recommender(data_dir=data_dir, models_dir=models_dir)
        
    interaction_df = joblib.load(interaction_path)
    similarity_df = joblib.load(sim_path)
    
    df_customers = pd.read_csv(cust_path)
    cust_row = df_customers[df_customers["customer_id"] == customer_id]
    
    if cust_row.empty:
        # Fallback to general popular services
        return get_default_recommendations(top_n=top_n)
        
    cust_info = cust_row.iloc[0]
    client_type = cust_info.get("client_type", "Independent Musician")
    primary_service = cust_info.get("primary_service", "Recording")
    segment = cust_info.get("segment_name", "Potential / Promising")
    
    # Customer's historical usage vector
    if customer_id in interaction_df.index:
        user_vector = interaction_df.loc[customer_id]
        used_services = user_vector[user_vector > 0].index.tolist()
    else:
        used_services = [primary_service]
        
    scores = {}
    reasons = {}
    
    for candidate in ALL_SERVICES:
        if candidate in used_services:
            # Service already booked before, slightly discount score unless it's a recurring service
            recurring_boost = 0.5 if candidate in ["Recording", "Rehearsal Space"] else 0.1
            scores[candidate] = recurring_boost
            reasons[candidate] = f"Frequent staple for repeat sessions ({candidate})"
        else:
            # Compute collaborative similarity score
            sim_score = 0.0
            for used in used_services:
                if used in similarity_df.columns and candidate in similarity_df.index:
                    sim_score += similarity_df.loc[candidate, used]
                    
            # Persona bonus
            persona_bonus = 0.0
            if "Musician" in client_type and candidate in ["Mixing", "Mastering", "Rehearsal Space"]:
                persona_bonus += 0.35
            elif "Director" in client_type and candidate in ["Music Production", "Mastering"]:
                persona_bonus += 0.40
            elif "Podcaster" in client_type and candidate in ["Podcast Recording", "Mixing"]:
                persona_bonus += 0.45
            elif "Voice" in client_type and candidate in ["Voiceover / Dubbing", "Recording"]:
                persona_bonus += 0.45
                
            total_score = sim_score + persona_bonus
            scores[candidate] = total_score
            reasons[candidate] = (
                f"Highly paired with your past '{primary_service}' bookings & popular among {client_type}s"
            )
            
    # Sort top N
    sorted_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    results = []
    for service_name, score in sorted_recs:
        results.append({
            "service": service_name,
            "match_score": round(min(score * 45 + 50, 99.5), 1), # Scale to realistic %
            "description": SERVICE_DESCRIPTIONS.get(service_name, ""),
            "rationale": reasons.get(service_name, "Recommended based on user similarity")
        })
        
    return results

def get_default_recommendations(primary_service="Recording", client_type="Independent Musician", top_n=3):
    """Fallback / Cold start recommendations for new clients."""
    sim_path = "models/service_similarity.pkl"
    if os.path.exists(sim_path):
        similarity_df = joblib.load(sim_path)
        if primary_service in similarity_df.columns:
            top_similar = similarity_df[primary_service].drop(labels=[primary_service]).sort_values(ascending=False).head(top_n)
            results = []
            for s, score in top_similar.items():
                results.append({
                    "service": s,
                    "match_score": round(float(score * 40 + 55), 1),
                    "description": SERVICE_DESCRIPTIONS.get(s, ""),
                    "rationale": f"Frequently booked together with {primary_service} by studio artists"
                })
            return results
            
    # Hardcoded top staples if no model yet
    return [
        {"service": "Mixing", "match_score": 94.5, "description": SERVICE_DESCRIPTIONS["Mixing"], "rationale": "Industry-standard next step after studio recording"},
        {"service": "Mastering", "match_score": 88.0, "description": SERVICE_DESCRIPTIONS["Mastering"], "rationale": "Final polish for streaming and commercial release"},
        {"service": "Rehearsal Space", "match_score": 81.2, "description": SERVICE_DESCRIPTIONS["Rehearsal Space"], "rationale": "High-volume studio prep for upcoming recordings"}
    ]

if __name__ == "__main__":
    train_and_save_recommender()
    sample_rec = get_recommendations_for_customer("CUST-1001")
    print("\nSample Recommendation for CUST-1001:")
    for r in sample_rec:
        print(f"- {r['service']} ({r['match_score']}%) : {r['rationale']}")
