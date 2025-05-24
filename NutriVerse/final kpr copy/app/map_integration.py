import folium
from folium.plugins import MarkerCluster

def plot_map(hotel_lat, hotel_lon, ashrams, volunteers):
    map_obj = folium.Map(location=[hotel_lat, hotel_lon], zoom_start=13)
    
    # Mark hotel location
    folium.Marker(
        [hotel_lat, hotel_lon],
        popup="📍 Donor Location (You)",
        icon=folium.Icon(color="blue")
    ).add_to(map_obj)
    
    # Mark Ashrams
    ashram_cluster = MarkerCluster(name="Ashrams").add_to(map_obj)
    for ashram in ashrams:
        folium.Marker(
            [ashram['latitude'], ashram['longitude']],
            popup=f"Ashram: {ashram['name']}",
            icon=folium.Icon(color="green", icon="home")
        ).add_to(ashram_cluster)

    # Mark Volunteers
    volunteer_cluster = MarkerCluster(name="Volunteers").add_to(map_obj)
    for volunteer in volunteers:
        folium.Marker(
            [volunteer['latitude'], volunteer['longitude']],
            popup=f"Volunteer: {volunteer['name']}",
            icon=folium.Icon(color="red", icon="user")
        ).add_to(volunteer_cluster)

    return map_obj



