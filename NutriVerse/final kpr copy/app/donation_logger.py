import pandas as pd
from datetime import datetime

def log_donation(hotel, food_type, ashram, volunteer):
    donations_df = pd.read_csv("data/donations.csv")

    new_entry = {
        "hotel": hotel,
        "food_type": food_type,
        "ashram": ashram,
        "volunteer": volunteer if volunteer else "None",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    donations_df = donations_df.append(new_entry, ignore_index=True)
    donations_df.to_csv("data/donations.csv", index=False)

def load_donations():
    return pd.read_csv("data/donations.csv")
