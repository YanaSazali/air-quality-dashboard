import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import re

st.set_page_config(page_title="Air Quality Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS styling
st.markdown("""
    <style>
    .health-alert {
        padding: 10px; border-radius: 5px; margin: 10px 0;
    }
    .good { background-color: #4CAF50; color: white; }
    .moderate { background-color: #FFEB3B; color: black; }
    .unhealthy-sensitive { background-color: #FF9800; color: white; }
    .unhealthy { background-color: #F44336; color: white; }
    .very-unhealthy { background-color: #9C27B0; color: white; }
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(0,0,0,.3);
        border-radius: 50%;
        border-top-color: #fff;
        animation: spin 1s ease-in-out infinite;
        margin-right: 10px;
    }
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    </style>
""", unsafe_allow_html=True)

# Load default dataset
@st.cache_resource
def load_default_data():
    try:
        df = pd.read_csv("AirQuality_Final_Processed.csv")
        # Handle date conversion
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            df['Date'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
            df.dropna(subset=['Date'], inplace=True)
        else:
            df['Date'] = pd.NaT
        return df
    except Exception as e:
        st.error(f"Error loading default data: {e}")
        return pd.DataFrame()

default_df = load_default_data()

# File uploader in sidebar
uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])

# Function to standardize column names
def standardize_columns(df):
    new_columns = {}
    for col in df.columns:
        lower_col = col.lower()
        # Standardize pollutant names
        if re.search(r'pm2\.?5', lower_col):
            new_columns[col] = 'PM2.5'
        elif re.search(r'pm10', lower_col):
            new_columns[col] = 'PM10'
        elif re.search(r'no2', lower_col):
            new_columns[col] = 'NO2'
        elif re.search(r'so2', lower_col):
            new_columns[col] = 'SO2'
        elif re.search(r'^co\b', lower_col) or re.search(r'carbon.?monoxide', lower_col):
            new_columns[col] = 'CO'
        elif re.search(r'o3', lower_col) or re.search(r'ozone', lower_col):
            new_columns[col] = 'O3'
        elif re.search(r'temp', lower_col):
            new_columns[col] = 'Temperature'
        elif re.search(r'humidity', lower_col):
            new_columns[col] = 'Humidity'
        elif re.search(r'wind', lower_col):
            new_columns[col] = 'Wind Speed'
        elif re.search(r'country', lower_col):
            new_columns[col] = 'Country'
        elif re.search(r'city', lower_col):
            new_columns[col] = 'City'
        elif re.search(r'date', lower_col) or re.search(r'time', lower_col):
            new_columns[col] = 'Date'
        else:
            new_columns[col] = col
    return df.rename(columns=new_columns)

# Function to load uploaded data
@st.cache_resource
def load_uploaded_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        df = standardize_columns(df)
        # Handle date conversion
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.dropna(subset=['Date'], inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading uploaded data: {e}")
        return pd.DataFrame()

# Show default dataset info in sidebar
with st.sidebar.expander("ℹ️ Default Dataset Info"):
    st.success("✅ Uploaded dataset used." if uploaded_file else "📁 Default dataset used.")
    st.write(f"Records: {default_df.shape[0]} | Columns: {default_df.shape[1]}")
    if st.checkbox("Show column names"):
        st.write(default_df.columns.tolist())

# Page navigation
page = st.sidebar.selectbox("Select Page", ["Home", "Dashboard", "Prediction", "Policy Simulation"])

if page == "Home":
    st.markdown("""
        <h1 style='text-align: center; color: #FF4B4B;'>Welcome to the Air Quality Dashboard</h1>
        <h4 style='text-align: center; color: gray;'>Track, analyze, and predict air quality for healthier communities</h4>
        <br>
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
            <h3>Key Features:</h3>
            <ul>
                <li><b>Dashboard:</b> Visualize air quality trends</li>
                <li><b>Prediction:</b> Estimate PM2.5 using ML models</li>
                <li><b>Policy Simulation:</b> Assess pollution reduction impact</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

elif page == "Dashboard":
    # Use uploaded_df if available, otherwise default_df
    df = load_uploaded_data(uploaded_file) if uploaded_file is not None else default_df
    
    # Show uploaded dataset preview on main page
    if uploaded_file is not None:
        with st.expander("📊 Uploaded Dataset Preview", expanded=True):
            st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            st.dataframe(df.head())
    
    st.markdown("## 🌍 Air Quality Dashboard")
    
    available_cols = df.columns.tolist()
    has_country = 'Country' in available_cols
    has_city = 'City' in available_cols
    
    # Standard pollutant columns we'll look for
    pollutant_options = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    available_pollutants = [col for col in pollutant_options if col in available_cols]
    
    if has_country and has_city:
        countries = df['Country'].dropna().unique()
        country = st.sidebar.selectbox("Select Country", sorted(countries) if len(countries) > 0 else ["No countries available"])
        
        if len(countries) > 0:
            cities = df[df['Country'] == country]['City'].dropna().unique()
            selected_cities = st.sidebar.multiselect("Select Cities", sorted(cities) if len(cities) > 0 else ["No cities available"])
        else:
            selected_cities = []
    else:
        country = None
        selected_cities = []
    
    pollutant = st.sidebar.selectbox("Select Pollutant", available_pollutants if available_pollutants else ["No pollutants available"])
    
    if (not has_country or not has_city or len(selected_cities) == 0 or not available_pollutants):
        st.info("Please select a country and city with available pollutant data")
    else:
        filtered_df = df[(df['Country'] == country) & (df['City'].isin(selected_cities))]
        
        st.markdown("### 📍 City Locations (if coordinates available)")
        try:
            # Create mock coordinates if none exist
            unique_cities = filtered_df['City'].unique()
            city_coords = {}
            for i, city in enumerate(unique_cities):
                city_coords[city] = (40 + i*5, -100 + i*10)  # Mock coordinates
            
            map_df = pd.DataFrame({
                'City': selected_cities,
                'Lat': [city_coords.get(city, (0, 0))[0] for city in selected_cities],
                'Lon': [city_coords.get(city, (0, 0))[1] for city in selected_cities],
                pollutant: [filtered_df[filtered_df['City'] == city][pollutant].mean() for city in selected_cities]
            })
            
            fig = px.scatter_mapbox(map_df, lat="Lat", lon="Lon", size=pollutant, 
                                  hover_name="City", color=pollutant,
                                  zoom=3, height=300, 
                                  color_continuous_scale=px.colors.sequential.Viridis)
            fig.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not display map: {str(e)}")
        
        st.markdown(f"### 📈 {pollutant} Over Time")
        try:
            fig, ax = plt.subplots(figsize=(10, 4))
            for city in selected_cities:
                city_data = filtered_df[filtered_df['City'] == city]
                if 'Date' in city_data.columns:
                    series = city_data.groupby('Date')[pollutant].mean()
                    ax.plot(series.index, series.values, label=city)
            ax.set_xlabel("Date")
            ax.set_ylabel(f"{pollutant} (µg/m³)")
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not create time series plot: {str(e)}")
        
        if 'PM2.5' in available_cols:
            st.markdown("### ⚠️ Health Alerts (Based on PM2.5)")
            for city in selected_cities:
                try:
                    avg = filtered_df[filtered_df['City'] == city]['PM2.5'].mean()
                    status, msg, css = interpret_pm25(avg)
                    st.markdown(f"<div class='health-alert {css}'><b>{city}: {status}</b> – {msg} (Avg PM2.5: {avg:.1f})</div>", unsafe_allow_html=True)
                except:
                    pass

elif page == "Prediction":
    # Always use default_df for prediction
    df = default_df
    available_cols = df.columns.tolist()
    
    if 'PM2.5' not in available_cols:
        st.info("PM2.5 prediction is not available with the current dataset.")
    else:
        st.markdown("## Predict PM2.5")
        model_name = st.selectbox("Model", ["Linear Regression", "Random Forest", "Decision Tree", "Neural Network"])
        
        # Create sliders for available features only
        slider_values = {}
        if 'PM10' in available_cols:
            slider_values['PM10'] = st.slider("PM10", 0.0, 200.0, float(df['PM10'].mean()))
        if 'NO2' in available_cols:
            slider_values['NO2'] = st.slider("NO2", 0.0, 100.0, float(df['NO2'].mean()))
        if 'CO' in available_cols:
            slider_values['CO'] = st.slider("CO", 0.0, 10.0, float(df['CO'].mean()))
        if 'O3' in available_cols:
            slider_values['O3'] = st.slider("O3", 0.0, 200.0, float(df['O3'].mean()))
        if 'Temperature' in available_cols:
            slider_values['Temperature'] = st.slider("Temperature", -10.0, 40.0, float(df['Temperature'].mean()))
        if 'Humidity' in available_cols:
            slider_values['Humidity'] = st.slider("Humidity", 0.0, 100.0, float(df['Humidity'].mean()))
        if 'Wind Speed' in available_cols:
            slider_values['Wind Speed'] = st.slider("Wind Speed", 0.0, 20.0, float(df['Wind Speed'].mean()))
        if 'City_Mean_PM25' in available_cols:
            slider_values['City_Mean_PM25'] = st.slider("City Avg PM2.5", 0.0, 150.0, float(df['City_Mean_PM25'].mean()))
        
        features = {}
        for col, val in slider_values.items():
            features[col] = val
        
        # Add derived features if possible
        if 'PM10' in features:
            features['PM_Ratio'] = features['PM10'] / (features['PM10'] * 0.5 + 1e-5)
        if 'Humidity' in features and 'Temperature' in features:
            features['Humidity_Temp'] = features['Humidity'] * features['Temperature']
        if 'O3' in features and 'NO2' in features:
            features['O3_NO2'] = features['O3'] / (features['NO2'] + 1e-5)
        
        # Add temporal features
        features.update({
            'Month': 6, 'Day': 15, 'Weekday': 2, 'Is_Weekend': 0,
            'Lag_PM2.5': features.get('City_Mean_PM25', 0) * 0.9,
            'Rolling_PM2.5': features.get('City_Mean_PM25', 0)
        })
        
        input_df = pd.DataFrame([features])
        model_features = [col for col in input_df.columns if col in available_cols]
        
        if len(model_features) > 0:
            model_map = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Neural Network": MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, early_stopping=True, random_state=42)
            }
            
            try:
                pipeline = Pipeline([("scaler", StandardScaler()), ("model", model_map[model_name])])
                pipeline.fit(df[model_features], df['PM2.5'])
                prediction = pipeline.predict(input_df[model_features])[0]
                aqi = calculate_aqi(prediction)
                status, msg, css = interpret_pm25(prediction)
                
                st.metric("Predicted PM2.5", f"{prediction:.1f} µg/m³")
                st.metric("AQI", f"{aqi:.0f}")
                st.markdown(f"<div class='health-alert {css}'><b>{status}</b> – {msg}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.info("Prediction couldn't be completed with the available data.")
        else:
            st.info("Not enough features available for prediction.")

elif page == "Policy Simulation":
    # Always use default_df for policy simulation
    df = default_df
    st.markdown("## Policy Simulation")
    available_cols = df.columns.tolist()
    sim_cols = [col for col in ['PM10', 'NO2', 'CO', 'O3'] if col in available_cols]
    
    if len(sim_cols) == 0:
        st.info("Policy simulation is not available with the current dataset.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            if 'PM10' in sim_cols:
                pm10_base = st.number_input("PM10", value=float(df['PM10'].mean()))
            if 'NO2' in sim_cols:
                no2_base = st.number_input("NO2", value=float(df['NO2'].mean()))
        with col2:
            if 'CO' in sim_cols:
                co_base = st.number_input("CO", value=float(df['CO'].mean()))
            if 'O3' in sim_cols:
                o3_base = st.number_input("O3", value=float(df['O3'].mean()))
        
        st.markdown("### Adjustment (%)")
        adjustments = {}
        if 'PM10' in sim_cols:
            adjustments['PM10'] = st.slider("PM10 Change", -50, 50, 0)
        if 'NO2' in sim_cols:
            adjustments['NO2'] = st.slider("NO2 Change", -50, 50, 0)
        if 'CO' in sim_cols:
            adjustments['CO'] = st.slider("CO Change", -50, 50, 0)
        if 'O3' in sim_cols:
            adjustments['O3'] = st.slider("O3 Change", -50, 50, 0)
        
        if st.button("Simulate Impact"):
            base_vals = {}
            if 'PM10' in sim_cols:
                base_vals['PM10'] = pm10_base
            if 'NO2' in sim_cols:
                base_vals['NO2'] = no2_base
            if 'CO' in sim_cols:
                base_vals['CO'] = co_base
            if 'O3' in sim_cols:
                base_vals['O3'] = o3_base
            
            new_vals = simulate_policy_change(base_vals, adjustments)
            
            try:
                model = RandomForestRegressor(random_state=42)
                model.fit(df[sim_cols], df['PM2.5'])
                
                baseline_input = [base_vals.get(col, 0) for col in sim_cols]
                new_input = [new_vals.get(col, 0) for col in sim_cols]
                
                baseline = model.predict([baseline_input])[0]
                new_pred = model.predict([new_input])[0]
                
                st.metric("Baseline PM2.5", f"{baseline:.1f}")
                st.metric("Simulated PM2.5", f"{new_pred:.1f}", delta=f"{new_pred - baseline:.1f}")
                if new_pred < baseline:
                    st.success(f"✅ Policy reduces PM2.5 by {(baseline - new_pred):.1f}")
                else:
                    st.warning(f"⚠️ Policy increases PM2.5 by {(new_pred - baseline):.1f}")
            except:
                st.info("Simulation couldn't be completed with the available data.")
