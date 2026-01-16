import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim

from geopy.geocoders import Nominatim
import time

from geopy.geocoders import ArcGIS

def get_coordinates(address):
    """
    Converts a string address to (lat, lon) using ArcGIS.
    Forced to NYC context to prevent global search errors.
    """
    try:
        # 1. Clean and focus the query
        # If the user just types "Brooklyn", we force it to "Brooklyn, NYC, NY"
        if "New York" not in address and "NY" not in address:
            search_query = f"{address}, New York City, NY"
        else:
            search_query = address

        # 2. Use ArcGIS (often more robust than Nominatim for NYC)
        geolocator = ArcGIS(timeout=10)
        
        # 3. Geocode with NYC context
        location = geolocator.geocode(search_query)
        
        if location:
            # For debugging purposes, you can uncomment the line below to see what was found
            # print(f"Found: {location.address} at {location.latitude}, {location.longitude}")
            return location.latitude, location.longitude
        
        return None, None
        
    except Exception as e:
        print(f"Geocoding error for {address}: {e}")
        return None, None
    
# Add this to src/processing.py

def haversine_np(lon1, lat1, lon2, lat2):
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c  # 6371 is the radius of Earth in km
    return km


def manhattan_haversine(lat1, lon1, lat2, lon2):
    # 1. Distance along the Latitude (North-South)
    # We keep Longitude constant to measure just the vertical leg
    d_lat = haversine_np(lon1, lat1, lon1, lat2)

    # 2. Distance along the Longitude (East-West)
    # We keep Latitude constant to measure just the horizontal leg
    d_lon = haversine_np(lon1, lat1, lon2, lat1)

    m_dist_k=d_lat+d_lon
    # 3. Manhattan Distance is the sum of the two legs
    
    return m_dist_k


def prepare_input(data_dict):
    return pd.DataFrame([data_dict])