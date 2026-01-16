import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from src.processing import prepare_input, get_coordinates, manhattan_haversine

# 1. Load the pipeline
pipeline = joblib.load('models/trip_pipeline1.pkl')

# 2. Helper Function for Validation
def is_in_nyc(lat, lon):
    # Expanded bounding box to safely cover NYC boroughs
    return 40.47 <= lat <= 40.95 and -74.30 <= lon <= -73.65

st.title("🚖 NYC Real-Time Trip Predictor")

# 3. User Inputs
pickup_addr = st.text_input("Pickup Address", placeholder="e.g., Times Square")
dropoff_addr = st.text_input("Dropoff Address", placeholder="e.g., Brooklyn Bridge")

if st.button("Estimate Trip Time"):
    if not pickup_addr or not dropoff_addr:
        st.warning("Please enter both addresses.")
    else:
        with st.spinner('Geocoding and calculating ETA...'):
            # A. Get Coordinates
            p_lat, p_lon = get_coordinates(pickup_addr)
            d_lat, d_lon = get_coordinates(dropoff_addr)

            if p_lat and d_lat:
                # B. Validate NYC Location
                if not is_in_nyc(p_lat, p_lon) or not is_in_nyc(d_lat, d_lon):
                    st.error("One of the addresses is outside of NYC. Please be more specific (e.g., add 'New York, NY').")
                    # Display map anyway so user sees where the pin dropped
                    st.map(pd.DataFrame({'lat': [p_lat, d_lat], 'lon': [p_lon, d_lon]}))
                else:
                    # C. Auto-generate Time Features
                    now = datetime.now()
                    
                    # Corrected Rush Hour Logic (e.g., 7-10 AM or 4-8 PM)
                    current_hour = now.hour
                    is_rush_hour = 1 if (7 <= current_hour <= 14) or (16 <= current_hour <= 20) else 0
                    
                    # D. Feature Engineering
                    distance = manhattan_haversine(p_lat, p_lon, d_lat, d_lon)
                    
                    user_data = {
                        'pickup_latitude': p_lat,
                        'pickup_longitude': p_lon,
                        'dropoff_latitude': d_lat,
                        'dropoff_longitude': d_lon,
                        'manhattan_dist_km': distance,
                        'is_rush_hour': is_rush_hour,
                    }

                    # E. Predict
                    input_df = prepare_input(user_data)
                    pred = pipeline.predict(input_df)
                    time_duration = pred[0]  # Raw output (no log)

                    # F. Display Results (Inside the success block)
                    st.success("Prediction complete!")
                    st.metric("Estimated Duration", f"{time_duration:.1f} Mins")
                    
                    # Show Trip Map
                    map_data = pd.DataFrame({'lat': [p_lat, d_lat], 'lon': [p_lon, d_lon]})
                    st.map(map_data)
            else:
                st.error("Could not find the location. Try adding 'NYC' to the address.")