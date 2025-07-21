import pandas as pd
import numpy as np
import os
def simulate_mmm_data(start_date="2022-01-01", weeks=104, random_seed=42):
    np.random.seed(random_seed)
    
    # Weekly date range
    dates = pd.date_range(start=start_date, periods=weeks, freq="W")
    
    # Marketing channel spends (simulate some variation)
    tv_spend = np.random.normal(loc=10000, scale=4000, size=weeks).clip(min=1000)
    radio_spend = np.random.normal(loc=4000, scale=1500, size=weeks).clip(min=500)
    digital_spend = np.random.normal(loc=12000, scale=5000, size=weeks).clip(min=2000)
    influencer_spend = np.random.normal(loc=3000, scale=1000, size=weeks).clip(min=500)
    ooh_spend = np.random.normal(loc=6000, scale=2000, size=weeks).clip(min=1000)
    
    # Simulated seasonality using a sine wave + noise
    seasonality_index = 100 * np.sin(np.linspace(0, 4 * np.pi, weeks)) + np.random.normal(0, 10, weeks)

    # Holiday flag — 0 for most weeks, 1 for holiday weeks
    holidays = np.zeros(weeks)
    holiday_weeks = np.random.choice(range(weeks), size=10, replace=False)
    holidays[holiday_weeks] = 1

    # Competitor activity: small random fluctuations
    competitor_activity = np.random.normal(loc=1000, scale=300, size=weeks).clip(min=500)

    # Generate Sales using a formula with noise
    base_sales = (
        0.04 * tv_spend + 
        0.05 * radio_spend +
        0.07 * digital_spend + 
        0.03 * influencer_spend +
        0.02 * ooh_spend + 
        0.5 * holidays + 
        0.3 * seasonality_index -
        0.03 * competitor_activity
    )
    noise = np.random.normal(0, 2000, weeks)
    sales = (base_sales + noise).clip(min=10000)

    # Combine into DataFrame
    df = pd.DataFrame({
        "Week": dates,
        "Sales": sales.round(2),
        "TV_Spend": tv_spend.round(2),
        "Radio_Spend": radio_spend.round(2),
        "Digital_Spend": digital_spend.round(2),
        "Influencer_Spend": influencer_spend.round(2),
        "OOH_Spend": ooh_spend.round(2),
        "Seasonality_Index": seasonality_index.round(2),
        "Holiday_Flag": holidays.astype(int),
        "Competitor_Activity": competitor_activity.round(2),
    })

    return df

if __name__ == "__main__":
    df = simulate_mmm_data()
    os.makedirs("./data/simulated", exist_ok=True)
    df.to_csv("./data/simulated/marketing_mmm_dataset.csv", index=False)
    print("Simulated MMM dataset saved to ./data/simulated/marketing_mmm_dataset.csv")
