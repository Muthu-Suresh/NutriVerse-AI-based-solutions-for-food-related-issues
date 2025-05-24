from geopy.distance import geodesic

def find_nearest_ashram(hotel_lat, hotel_lon, ashrams_data, food_type, preferred_time, total_people):
    nearest = None
    min_distance = float('inf')

    for ashram in ashrams_data:
        try:
            if (
                food_type in ashram['food_types'].split(',') and
                preferred_time == ashram['preferred_times'] and
                total_people <= int(ashram['max_capacity'])
            ):
                ashram_location = (ashram['latitude'], ashram['longitude'])
                distance = geodesic((hotel_lat, hotel_lon), ashram_location).km

                if distance < min_distance:
                    min_distance = distance
                    nearest = ashram
        except KeyError as e:
            print(f"Missing key in ashram data: {e}")
            continue

    return nearest


