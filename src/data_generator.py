import os
import numpy as np
import pandas as pd

def generate_telecom_data(num_customers=5000, seed=42):
    np.random.seed(seed)
    
    # 1. Customer IDs
    customer_ids = [f"NP-{i:05d}" for i in range(1, num_customers + 1)]
    
    # 2. Demographics
    age = np.random.randint(18, 76, size=num_customers)
    gender = np.random.choice(["Male", "Female", "Other"], size=num_customers, p=[0.51, 0.47, 0.02])
    
    provinces = ["Koshi", "Madhesh", "Bagmati", "Gandaki", "Lumbini", "Karnali", "Sudurpashchim"]
    province_probs = [0.18, 0.20, 0.22, 0.10, 0.16, 0.06, 0.08]
    province = np.random.choice(provinces, size=num_customers, p=province_probs)
    
    district_types = ["Urban", "Semi-Urban", "Rural"]
    district_type = np.random.choice(district_types, size=num_customers, p=[0.45, 0.35, 0.20])
    
    sim_types = ["Prepaid", "Postpaid"]
    sim_type = np.random.choice(sim_types, size=num_customers, p=[0.90, 0.10])
    
    tenure_days = np.random.randint(15, 1801, size=num_customers) # up to ~5 years
    
    # 3. Usage features
    calls_min_30d = np.random.gamma(shape=3.0, scale=80.0, size=num_customers).round(1) # mean 240 mins
    sms_count_30d = np.random.poisson(lam=45, size=num_customers)
    data_gb_30d = np.random.lognormal(mean=2.2, sigma=0.8, size=num_customers).round(2) # mean ~12 GB
    # Clip data usage to realistic limits (max 150GB)
    data_gb_30d = np.clip(data_gb_30d, 0.0, 150.0)
    
    night_usage_pct = np.random.beta(a=2, b=5, size=num_customers) * 100 # mean ~28%
    
    # 4. Recharge behavior
    last_recharge_days_ago = np.random.geometric(p=0.08, size=num_customers) - 1 # mean ~11 days
    last_recharge_days_ago = np.clip(last_recharge_days_ago, 0, 90)
    
    # Average recharge amount centered around typical values: 100, 200, 500, 1000 NPR
    base_recharges = [50, 100, 200, 500, 1000, 1500]
    avg_recharge_amount_npr = np.random.choice(base_recharges, size=num_customers, p=[0.1, 0.4, 0.25, 0.15, 0.07, 0.03])
    # Add minor noise to make it realistic
    avg_recharge_amount_npr = avg_recharge_amount_npr + np.random.randint(-15, 15, size=num_customers)
    avg_recharge_amount_npr = np.clip(avg_recharge_amount_npr, 20, 2000).astype(float).round(1)
    
    recharge_count_30d = np.random.poisson(lam=3, size=num_customers)
    recharge_count_30d = np.where(sim_type == "Postpaid", np.random.choice([1, 2], size=num_customers), recharge_count_30d)
    
    # 5. Network Quality
    # Signal strength in dBm (-120 is poor, -50 is excellent)
    signal_strength_dbm = np.random.randint(-115, -49, size=num_customers)
    
    # Call drop rate is correlated with signal strength (poorer signal -> higher drop rate)
    # Norm signal to 0-1 (0 is excellent, 1 is worst)
    norm_signal = (-signal_strength_dbm - 50) / 65.0
    call_drop_rate = (np.random.beta(a=1, b=8, size=num_customers) * 0.15 + norm_signal * 0.05).round(4)
    call_drop_rate = np.clip(call_drop_rate, 0.0, 0.25)
    
    # Average data speed (poorer signal -> lower speed)
    avg_data_speed_mbps = (np.random.lognormal(mean=2.8, sigma=0.6, size=num_customers) * (1 - norm_signal * 0.7)).round(2)
    avg_data_speed_mbps = np.clip(avg_data_speed_mbps, 0.1, 120.0)
    
    # 6. Complaints
    # Rural areas and bad signal -> more complaints
    complaint_prob = 0.05 + 0.15 * norm_signal + 0.05 * (district_type == "Rural")
    has_complaint = np.random.rand(num_customers) < complaint_prob
    num_complaints_30d = np.where(has_complaint, np.random.poisson(lam=1.5, size=num_customers) + 1, 0)
    
    avg_resolution_time_hours = np.where(num_complaints_30d > 0, 
                                          np.random.gamma(shape=3.0, scale=8.0, size=num_customers).round(1), 
                                          0.0)
    
    # 7. Service packs
    data_pack_active = np.random.choice([0, 1], size=num_customers, p=[0.4, 0.6])
    voice_pack_active = np.random.choice([0, 1], size=num_customers, p=[0.6, 0.4])
    vas_active = np.random.choice([0, 1], size=num_customers, p=[0.8, 0.2])
    roaming_active = np.random.choice([0, 1], size=num_customers, p=[0.95, 0.05])
    
    # 8. Derived Features
    # usage_drop_pct: drop in usage compared to last month (higher is worse)
    usage_drop_pct = np.random.normal(loc=0.15, scale=0.25, size=num_customers).round(3)
    # Correlate usage drop with signal strength and complaints
    usage_drop_pct += (norm_signal * 0.15 + (num_complaints_30d > 0) * 0.1)
    usage_drop_pct = np.clip(usage_drop_pct, -0.5, 1.0)
    
    # recharge_drop_pct
    recharge_drop_pct = np.random.normal(loc=0.10, scale=0.20, size=num_customers).round(3)
    recharge_drop_pct += (usage_drop_pct * 0.3)
    recharge_drop_pct = np.clip(recharge_drop_pct, -0.5, 1.0)
    
    # inactive_days: correlated with recharge drop and usage drop
    inactive_days = np.random.poisson(lam=2, size=num_customers)
    inactive_days = np.where(usage_drop_pct > 0.5, inactive_days + np.random.randint(5, 15, size=num_customers), inactive_days)
    inactive_days = np.where(last_recharge_days_ago > 30, inactive_days + np.random.randint(3, 10, size=num_customers), inactive_days)
    inactive_days = np.clip(inactive_days, 0, 30)
    
    # 9. Target: Churn
    # We define a logit function based on the features to make model training realistic
    logit = (
        -3.2 # baseline logit
        + 3.5 * call_drop_rate
        + 0.5 * num_complaints_30d
        + 0.01 * avg_resolution_time_hours
        + 2.0 * usage_drop_pct
        + 1.8 * recharge_drop_pct
        + 0.12 * inactive_days
        + 0.03 * last_recharge_days_ago
        + 2.2 * norm_signal # Poor network signal increases churn
        - 0.001 * tenure_days # Loyal long-term customers churn less
        - 0.5 * data_pack_active
        - 0.3 * voice_pack_active
        - 0.0005 * avg_recharge_amount_npr # High spenders churn less
        - 0.6 * (sim_type == "Postpaid") # Postpaid customers churn less
    )
    
    # Convert logit to probability
    churn_prob = 1 / (1 + np.exp(-logit))
    # Add small noise to probability
    churn_prob = np.clip(churn_prob + np.random.normal(0, 0.05, size=num_customers), 0.0, 1.0)
    
    # Generate binary churn outcome
    churn = np.random.binomial(1, churn_prob)
    
    # Assembly
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": age,
        "gender": gender,
        "province": province,
        "district_type": district_type,
        "sim_type": sim_type,
        "tenure_days": tenure_days,
        "calls_min_30d": calls_min_30d,
        "sms_count_30d": sms_count_30d,
        "data_gb_30d": data_gb_30d,
        "night_usage_pct": night_usage_pct.round(2),
        "last_recharge_days_ago": last_recharge_days_ago,
        "avg_recharge_amount_npr": avg_recharge_amount_npr,
        "recharge_count_30d": recharge_count_30d,
        "signal_strength_dbm": signal_strength_dbm,
        "call_drop_rate": call_drop_rate,
        "avg_data_speed_mbps": avg_data_speed_mbps,
        "num_complaints_30d": num_complaints_30d,
        "avg_resolution_time_hours": avg_resolution_time_hours,
        "data_pack_active": data_pack_active,
        "voice_pack_active": voice_pack_active,
        "vas_active": vas_active,
        "roaming_active": roaming_active,
        "usage_drop_pct": usage_drop_pct.round(3),
        "recharge_drop_pct": recharge_drop_pct.round(3),
        "inactive_days": inactive_days,
        "churn": churn
    })
    
    return df

if __name__ == "__main__":
    # Create directory if it does not exist
    os.makedirs("data", exist_ok=True)
    
    df = generate_telecom_data(num_customers=5000)
    output_path = os.path.join("data", "telecom_churn_nepal.csv")
    df.to_csv(output_path, index=False)
    print(f"Dataset generated and saved successfully to {output_path}")
    print(f"Dataset Shape: {df.shape}")
    print(f"Churn Distribution:\n{df['churn'].value_counts(normalize=True)}")
