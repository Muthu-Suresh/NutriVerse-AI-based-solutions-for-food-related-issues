import pandas as pd
from app.haversine import haversine

def assign_volunteer(hotel_lat, hotel_lon):
    volunteers_df = pd.read_csv("data/volunteers.csv")
    volunteers_df = volunteers_df[volunteers_df['available'] == "Yes"]

    if volunteers_df.empty:
        return None

    volunteers_df['distance'] = volunteers_df.apply(
        lambda row: haversine(hotel_lat, hotel_lon, row['latitude'], row['longitude']), axis=1)

    nearest_volunteer = volunteers_df.loc[volunteers_df['distance'].idxmin()]

    return nearest_volunteer
