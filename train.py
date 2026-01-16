import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load data
df = pd.read_csv('data/final_df_modeling.csv')


# 1. Prepare Target (The log transform we discussed)
y = df['trip_duration_mt']
X = df.drop(columns=['trip_duration_mt', 'traffic_density_numeric'])

# 2. Define Preprocessing for different column types
numeric_features = [
    'pickup_latitude', 'pickup_longitude', 
    'dropoff_latitude', 'dropoff_longitude', 
    'manhattan_dist_km'
]

categorical_features = ['is_rush_hour']

preprocessor = ColumnTransformer(
    transformers=[
        # StandardScaler handles all numeric values including density
        ('num', StandardScaler(), numeric_features),
        # OneHotEncoder handles the binary/nominal flags
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 3. Create the full Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(objective='reg:squarederror', n_estimators=100))
])

# 4. Fit and Save
model_pipeline.fit(X, y)
joblib.dump(model_pipeline, 'models/trip_pipeline1.pkl')