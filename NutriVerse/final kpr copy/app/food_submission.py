import pandas as pd
import os

def save_donation(donation_data, file_path='data/sample_donations.csv'):
    df = pd.DataFrame([donation_data])

    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, mode='w', header=True, index=False)
