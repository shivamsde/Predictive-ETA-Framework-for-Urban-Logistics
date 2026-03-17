In urban logistics, "Last-Mile" delivery accounts for nearly 28% of total transportation costs. This project addresses the ETA Variance Problem by moving beyond simple distance-based calculations. By leveraging the NYC Taxi Dataset (1M+ records), this framework predicts trip durations with high precision, accounting for urban morphology, temporal congestion patterns, and stochastic road variables.

🚀 Key Features
Feature Engineering Pipeline: Transformations for haversine distance, pickup/drop-off clusters, and cyclical temporal features (Hour/Day/Month).

Gradient Boosted Architecture: Utilizes XGBoost and Random Forest ensembles to capture non-linear traffic relationships.

Geospatial Intelligence: Integration of coordinate-based density mapping to identify "High-Latency Zones."

Live Deployment: A functional Streamlit dashboard for real-time "What-If" scenario analysis (e.g., How does a Friday at 6:00 PM affect delivery windows?)

🏗️ Technical Architecture
Data Ingestion: Processing high-volume parquet files (NYC TLC Dataset).

Preprocessing: Outlier detection (Speed/Duration/Distance) and coordinate clipping.

Feature Evolution: * Temporal: Peak hour indicators, weekend binary flags.

Spatial: Neighborhood clustering using KMeans to identify logistics hubs.

Modeling: Hyperparameter tuning via GridSearchCV to minimize Mean Absolute Error (MAE).

Interface: Interactive UI for logistics dispatchers to simulate routes.

🛠️ Tech Stack
Language: Python

Data Ops: Pandas, NumPy, Scikit-learn

Modeling: XGBoost, Random Forest, Linear Regression

Visualization: Plotly, Matplotlib, Seaborn

Deployment: Streamlit

🏁 Getting Started
Clone the repository: git clone https://github.com/your-username/urban-logistics-eta.git

Install dependencies: pip install -r requirements.txt

Run the dashboard: streamlit run src/app.py
