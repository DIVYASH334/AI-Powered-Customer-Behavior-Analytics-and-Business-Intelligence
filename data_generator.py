"""
data_generator.py
Generates a realistic synthetic dataset for Sound/Recording Studio customer analytics.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

SERVICE_TYPES = [
    "Recording",
    "Mixing",
    "Mastering",
    "Rehearsal Space",
    "Voiceover / Dubbing",
    "Music Production",
    "Podcast Recording"
]

SERVICE_HOURLY_RATES = {
    "Recording": 1500,
    "Mixing": 1800,
    "Mastering": 2200,
    "Rehearsal Space": 800,
    "Voiceover / Dubbing": 1200,
    "Music Production": 2500,
    "Podcast Recording": 1000
}

CITIES = ["Chennai", "Coimbatore", "Bangalore", "Madurai", "Hyderabad", "Kochi", "Mumbai"]
CITY_WEIGHTS = [0.40, 0.15, 0.15, 0.10, 0.08, 0.07, 0.05]

FIRST_NAMES = [
    "Aravindh", "Karthik", "Divya", "Anirudh", "Vijay", "Siddharth", "Pooja", "Vikram",
    "Shreya", "Surya", "Meera", "Hari", "Sanjay", "Deepa", "Rahul", "Naveen", "Swathi",
    "Ashwin", "Priya", "Mano", "Arun", "Sneha", "Dhanush", "Keerthi", "Gautham", "Harini",
    "Kishore", "Nithya", "Vignesh", "Roshini", "Pradeep", "Lakshmi", "Kavya", "Suresh"
]

LAST_NAMES = [
    "Kumar", "Rajan", "Natarajan", "Ravichandran", "Subramanian", "Iyer", "Menon",
    "Sharma", "Varma", "Reddy", "Naidu", "Balaji", "Sundaram", "Krishnan", "Pillai", "Murugan"
]

PAYMENT_MODES = ["UPI", "Credit Card", "Debit Card", "Net Banking", "Cash"]
PAYMENT_WEIGHTS = [0.45, 0.25, 0.15, 0.10, 0.05]

STUDIO_ROOMS = ["Studio A (SSL Console)", "Studio B (Live Room)", "Vocal Booth Alpha", "Pod Suite 1", "Mix & Master Suite"]

def generate_customer_profiles(num_customers=1000):
    """Generate demographic profiles for unique customers."""
    customers = []
    
    for i in range(1, num_customers + 1):
        cust_id = f"CUST-{1000 + i}"
        name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        age = int(np.random.choice(
            [random.randint(18, 24), random.randint(25, 34), random.randint(35, 49), random.randint(50, 65)],
            p=[0.25, 0.45, 0.22, 0.08]
        ))
        gender = np.random.choice(["Male", "Female", "Non-Binary"], p=[0.60, 0.36, 0.04])
        city = np.random.choice(CITIES, p=CITY_WEIGHTS)
        
        client_type = np.random.choice(
            ["Independent Musician", "Band Member", "Film/Ad Music Director", "Podcaster", "Voice Actor", "Content Creator", "Corporate"],
            p=[0.30, 0.20, 0.15, 0.15, 0.10, 0.07, 0.03]
        )
        
        customers.append({
            "customer_id": cust_id,
            "name": name,
            "age": age,
            "gender": gender,
            "city": city,
            "client_type": client_type
        })
        
    return pd.DataFrame(customers)

def generate_bookings_dataset(df_customers, start_date="2024-01-01", end_date="2026-03-01"):
    """
    Generate realistic transactional bookings where each customer has between 1 and 12 visits,
    creating realistic repeat behavior and natural RFM clusters.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days
    
    bookings = []
    booking_counter = 5000
    
    # Assign visit frequencies: ~42% one-time, 58% repeat
    visit_options = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
    visit_probs =   [0.42, 0.22, 0.14, 0.08, 0.05, 0.04, 0.02, 0.015, 0.01, 0.005]
    
    for _, cust in df_customers.iterrows():
        cust_id = cust["customer_id"]
        client_type = cust["client_type"]
        num_visits = np.random.choice(visit_options, p=visit_probs)
        
        # Determine customer's base satisfaction / loyalty tier
        customer_satisfaction_bias = np.random.choice([5, 4, 3, 2], p=[0.50, 0.35, 0.10, 0.05])
        
        # Decide preferred service
        if client_type in ["Independent Musician", "Band Member"]:
            service_pool = ["Recording", "Mixing", "Rehearsal Space", "Mastering"]
            service_p = [0.45, 0.25, 0.20, 0.10]
        elif client_type == "Film/Ad Music Director":
            service_pool = ["Music Production", "Mixing", "Mastering", "Recording"]
            service_p = [0.40, 0.30, 0.20, 0.10]
        elif client_type == "Podcaster":
            service_pool = ["Podcast Recording", "Mixing", "Voiceover / Dubbing"]
            service_p = [0.70, 0.20, 0.10]
        elif client_type == "Voice Actor":
            service_pool = ["Voiceover / Dubbing", "Recording"]
            service_p = [0.80, 0.20]
        else:
            service_pool = SERVICE_TYPES
            service_p = [1/len(SERVICE_TYPES)] * len(SERVICE_TYPES)
            
        # Pick dates: sorted ascending
        day_offsets = sorted(random.sample(range(0, total_days), num_visits))
        
        for offset in day_offsets:
            booking_counter += 1
            booking_id = f"BK-{booking_counter}"
            booking_date = start_dt + timedelta(days=offset)
            
            service_type = np.random.choice(service_pool, p=service_p)
            room = random.choice(STUDIO_ROOMS)
            
            # Duration hours
            if service_type == "Rehearsal Space":
                duration = np.random.choice([2, 3, 4, 6], p=[0.35, 0.35, 0.20, 0.10])
            elif service_type in ["Recording", "Music Production"]:
                duration = np.random.choice([3, 4, 6, 8, 10], p=[0.25, 0.35, 0.25, 0.10, 0.05])
            elif service_type == "Podcast Recording":
                duration = np.random.choice([1, 2, 3], p=[0.40, 0.45, 0.15])
            else:
                duration = np.random.choice([2, 3, 4, 5], p=[0.30, 0.40, 0.20, 0.10])
                
            base_rate = SERVICE_HOURLY_RATES[service_type]
            base_amount = duration * base_rate
            
            addon_charge = np.random.choice([0, 500, 1000, 1500], p=[0.50, 0.30, 0.13, 0.07])
            discount_pct = np.random.choice([0, 5, 10, 15, 20], p=[0.60, 0.18, 0.12, 0.07, 0.03])
            discount_amount = (base_amount + addon_charge) * (discount_pct / 100.0)
            final_amount = round(base_amount + addon_charge - discount_amount, 2)
            
            # Rating with slight noise around customer satisfaction bias
            rating_draw = int(np.clip(round(customer_satisfaction_bias + np.random.normal(0, 0.5)), 1, 5))
            
            if rating_draw >= 4:
                feedback = random.choice(["Excellent acoustics & engineer", "Crisp sound quality", "Great vibe & gear", "Very professional session"])
            elif rating_draw == 3:
                feedback = random.choice(["Decent experience", "Good gear, slight AC hum", "Acceptable output"])
            else:
                feedback = random.choice(["Session delayed by 30 mins", "Monitoring headphone issue", "Overpriced for service"])
                
            payment_method = np.random.choice(PAYMENT_MODES, p=PAYMENT_WEIGHTS)
            
            bookings.append({
                "booking_id": booking_id,
                "customer_id": cust_id,
                "booking_date": booking_date.strftime("%Y-%m-%d"),
                "booking_month": booking_date.strftime("%Y-%m"),
                "day_of_week": booking_date.strftime("%A"),
                "service_type": service_type,
                "studio_room": room,
                "duration_hours": duration,
                "hourly_rate": base_rate,
                "addon_charge": addon_charge,
                "discount_percent": discount_pct,
                "amount_paid": final_amount,
                "payment_method": payment_method,
                "rating": rating_draw,
                "feedback": feedback
            })
            
    return pd.DataFrame(bookings)

def create_customer_summary(df_customers, df_bookings, reference_date="2026-03-05"):
    ref_dt = datetime.strptime(reference_date, "%Y-%m-%d")
    df_bookings["booking_dt"] = pd.to_datetime(df_bookings["booking_date"])
    
    agg_dict = {
        "booking_id": "count",
        "amount_paid": ["sum", "mean"],
        "duration_hours": ["sum", "mean"],
        "rating": "mean",
        "booking_dt": "max",
        "discount_percent": "mean"
    }
    
    grouped = df_bookings.groupby("customer_id").agg(agg_dict)
    grouped.columns = [
        "visit_count",
        "total_spend",
        "avg_booking_spend",
        "total_hours",
        "avg_duration_hours",
        "avg_rating",
        "last_visit_date",
        "avg_discount_received"
    ]
    grouped = grouped.reset_index()
    
    grouped["days_since_last_visit"] = (ref_dt - grouped["last_visit_date"]).dt.days
    grouped["days_since_last_visit"] = grouped["days_since_last_visit"].clip(lower=1)
    
    merged = pd.merge(df_customers, grouped, on="customer_id", how="left")
    
    merged["visit_count"] = merged["visit_count"].fillna(0).astype(int)
    merged["total_spend"] = merged["total_spend"].fillna(0)
    merged["avg_booking_spend"] = merged["avg_booking_spend"].fillna(0)
    merged["total_hours"] = merged["total_hours"].fillna(0)
    merged["avg_duration_hours"] = merged["avg_duration_hours"].fillna(0)
    merged["avg_rating"] = merged["avg_rating"].fillna(3.0).round(2)
    merged["avg_discount_received"] = merged["avg_discount_received"].fillna(0).round(1)
    merged["days_since_last_visit"] = merged["days_since_last_visit"].fillna(365).astype(int)
    merged["last_visit_date"] = merged["last_visit_date"].dt.strftime("%Y-%m-%d")
    
    top_services = df_bookings.groupby(["customer_id", "service_type"]).size().reset_index(name="count")
    top_services = top_services.sort_values(["customer_id", "count"], ascending=[True, False])
    top_services = top_services.drop_duplicates(subset=["customer_id"])
    top_services = top_services.rename(columns={"service_type": "primary_service"}).drop(columns=["count"])
    
    merged = pd.merge(merged, top_services, on="customer_id", how="left")
    merged["primary_service"] = merged["primary_service"].fillna("Recording")
    
    # Target 1: Repeat Customer (Yes / No) -> visit_count >= 2
    merged["repeat_customer"] = (merged["visit_count"] >= 2).astype(int)
    
    # Target 2: Churn Risk -> High recency or poor satisfaction
    def evaluate_churn(row):
        if row["days_since_last_visit"] > 160:
            return 1
        elif row["avg_rating"] < 3.0 and row["days_since_last_visit"] > 60:
            return 1
        elif row["visit_count"] == 1 and row["days_since_last_visit"] > 90:
            return 1
        else:
            return 0
            
    merged["churn_risk"] = merged.apply(evaluate_churn, axis=1)
    
    return merged

def main(data_dir="data"):
    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    print("-> Generating Customer Profiles...")
    df_customers = generate_customer_profiles(num_customers=1000)
    
    print("-> Generating Bookings Transactions...")
    df_bookings = generate_bookings_dataset(df_customers)
    
    print("-> Creating Aggregated Customer Summary Dataset...")
    df_customer_summary = create_customer_summary(df_customers, df_bookings)
    
    bookings_file = os.path.join(raw_dir, "sound_studio_bookings.csv")
    customers_file = os.path.join(raw_dir, "sound_studio_customers.csv")
    
    df_bookings.to_csv(bookings_file, index=False)
    df_customer_summary.to_csv(customers_file, index=False)
    
    print("Data Generation Completed!")
    print(f"Bookings saved: {len(df_bookings)} rows")
    print(f"Customers saved: {len(df_customer_summary)} rows")

if __name__ == "__main__":
    main()
