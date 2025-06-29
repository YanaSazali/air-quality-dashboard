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
@st.cache_data
def load_default_data():
    try:
        default_df = pd.read_csv("AirQuality_Final_Processed.csv")
        # Handle date conversion for default dataset
        date_cols = [col for col in default_df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            default_df['Date'] = pd.to_datetime(default_df[date_cols[0]], errors='coerce')
            default_df.dropna(subset=['Date'], inplace=True)
        return default_df
    except:
        return pd.DataFrame()

default_df = load_default_data()

# File uploader in sidebar
uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])

@st.cache_data
def load_uploaded_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        # Handle date conversion for uploaded data
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            df['Date'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
            df.dropna(subset=['Date'], inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading uploaded file: {e}")
        return None

# Standardize column names for pollutants
def standardize_columns(df):
    column_mapping = {
        'pm25': 'PM2.5',
        'pm2.5': 'PM2.5',
        'pm10': 'PM10',
        'no2': 'NO2',
        'so2': 'SO2',
        'co': 'CO',
        'o3': 'O3',
        'c0': 'CO',
        'temperature': 'Temperature',
        'humidity': 'Humidity',
        'wind speed': 'Wind Speed',
        'city': 'City',
        'country': 'Country'
    }
    
    df.columns = df.columns.str.lower()
    df.rename(columns=column_mapping, inplace=True)
    return df

# Determine which dataset to use
if uploaded_file:
    df = load_uploaded_data(uploaded_file)
    if df is not None:
        df = standardize_columns(df)
        current_df = df
        show_uploaded_data = True
    else:
        current_df = default_df
        show_uploaded_data = False
else:
    current_df = default_df
    show_uploaded_data = False

# Show dataset info in sidebar
with st.sidebar.expander("ℹ️ Dataset Info"):
    if uploaded_file and df is not None:
        st.success("✅ Uploaded dataset used.")
    else:
        st.info("📁 Default dataset used.")
    st.write(f"Records: {current_df.shape[0]} | Columns: {current_df.shape[1]}")
    
    if st.checkbox("Show default dataset columns"):
        st.write(default_df.columns.tolist())
    
    if uploaded_file and df is not None and st.checkbox("Show uploaded dataset columns"):
        st.write(df.columns.tolist())

# Show uploaded data preview on main page (not sidebar)
if show_uploaded_data:
    with st.expander("📊 Uploaded Dataset Preview", expanded=True):
        st.write(f"First 5 rows of your uploaded dataset (showing {min(5, len(df))} rows):")
        st.dataframe(df.head())

def calculate_aqi(pm25):
    try:
        if pm25 <= 12.0:
            return ((50-0)/(12.0-0)) * (pm25-0) + 0
        elif pm25 <= 35.4:
            return ((100-51)/(35.4-12.1)) * (pm25-12.1) + 51
        elif pm25 <= 55.4:
            return ((150-101)/(55.4-35.5)) * (pm25-35.5) + 101
        elif pm25 <= 150.4:
            return ((200-151)/(150.4-55.5)) * (pm25-55.5) + 151
        else:
            return ((300-201)/(250.4-150.5)) * (pm25-150.5) + 201
    except:
        return np.nan

def interpret_pm25(value):
    if value <= 12:
        return ("🟢 Good", "Air quality is satisfactory.", "good")
    elif value <= 35.4:
        return ("🟡 Moderate", "Acceptable for most, but sensitive groups may be affected.", "moderate")
    elif value <= 55.4:
        return ("🟠 Unhealthy for Sensitive Groups", "Sensitive individuals should reduce prolonged exertion.", "unhealthy-sensitive")
    elif value <= 150.4:
        return ("🔴 Unhealthy", "Everyone may experience health effects.", "unhealthy")
    else:
        return ("⚫ Very Unhealthy", "Health warnings for everyone.", "very-unhealthy")

def simulate_policy_change(base_values, adjustments):
    adjusted = base_values.copy()
    for k, v in adjustments.items():
        adjusted[k] = adjusted.get(k, 0) * (1 + v / 100)
    return adjusted

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
        <h3>💡 Did You Know?</h3>
        <blockquote>
            PM2.5 particles are smaller than a human hair and can enter your lungs.
        </blockquote>
        <h3>Understanding Pollutants</h3>
        <ul>
            <li><b>PM2.5:</b> Fine particulate matter (≤2.5 micrometers) that can penetrate deep into the lungs and even enter the bloodstream.</li>
            <li><b>PM10:</b> Coarse particulate matter (≤10 micrometers) that can cause respiratory irritation.</li>
            <li><b>NO₂ (Nitrogen Dioxide):</b> Emitted from vehicles and industrial activity; contributes to smog and lung irritation.</li>
            <li><b>SO₂ (Sulfur Dioxide):</b> Produced by burning fossil fuels; can cause respiratory issues and form acid rain.</li>
            <li><b>CO (Carbon Monoxide):</b> A colorless, odorless gas from incomplete combustion; dangerous at high levels.</li>
            <li><b>O₃ (Ozone):</b> Ground-level ozone is formed by chemical reactions in sunlight and can trigger breathing problems.</li>
        </ul>
        <p>Tracking these pollutants helps assess air quality and identify health risks in your area.</p>
    """, unsafe_allow_html=True)

elif page == "Dashboard":
    st.markdown("## 🌍 Air Quality Dashboard")
    
    # Use current_df (either uploaded or default)
    available_cols = current_df.columns.tolist()
    
    # Try to find location columns with flexible matching
    location_cols = {
        'country': [col for col in available_cols if 'country' in col.lower()],
        'city': [col for col in available_cols if 'city' in col.lower() or 'location' in col.lower()]
    }
    
    has_country = len(location_cols['country']) > 0
    has_city = len(location_cols['city']) > 0
    
    if has_country and has_city:
        country_col = location_cols['country'][0]
        city_col = location_cols['city'][0]
        
        countries = current_df[country_col].dropna().unique()
        country = st.sidebar.selectbox("Select Country", sorted(countries) if len(countries) > 0 else ["No countries available"])
        
        if len(countries) > 0:
            cities = current_df[current_df[country_col] == country][city_col].dropna().unique()
            selected_cities = st.sidebar.multiselect("Select Cities", sorted(cities) if len(cities) > 0 else ["No cities available"])
        else:
            selected_cities = []
    else:
        country = None
        selected_cities = []
    
    # Flexible pollutant detection
    pollutant_mapping = {
        'PM2.5': ['pm25', 'pm2.5', 'pm2_5'],
        'PM10': ['pm10'],
        'NO2': ['no2', 'nitrogen dioxide'],
        'SO2': ['so2', 'sulfur dioxide'],
        'CO': ['co', 'carbon monoxide', 'c0'],
        'O3': ['o3', 'ozone']
    }
    
    available_pollutants = []
    for standard_name, variants in pollutant_mapping.items():
        for variant in variants:
            if variant in [col.lower() for col in available_cols]:
                available_pollutants.append(standard_name)
                break
    
    pollutant = st.sidebar.selectbox("Select Pollutant", available_pollutants if available_pollutants else ["No pollutants available"])
    
    # Get actual column name for selected pollutant
    actual_col = None
    if pollutant != "No pollutants available":
        for variant in pollutant_mapping[pollutant]:
            if variant in [col.lower() for col in available_cols]:
                actual_col = [col for col in available_cols if col.lower() == variant][0]
                break
    
    if has_country and has_city and len(selected_cities) > 0 and actual_col:
        filtered_df = current_df[(current_df[country_col] == country) & (current_df[city_col].isin(selected_cities))]
        
        st.markdown("### 📍 City Locations (if coordinates available)")
        try:
            # Try to find latitude/longitude columns
            lat_col = [col for col in available_cols if 'lat' in col.lower()][0]
            lon_col = [col for col in available_cols if 'lon' in col.lower() or 'long' in col.lower()][0]
            
            map_df = pd.DataFrame({
                'City': selected_cities,
                'Lat': [filtered_df[filtered_df[city_col] == city][lat_col].mean() for city in selected_cities],
                'Lon': [filtered_df[filtered_df[city_col] == city][lon_col].mean() for city in selected_cities],
                pollutant: [filtered_df[filtered_df[city_col] == city][actual_col].mean() for city in selected_cities]
            })
            fig = px.scatter_mapbox(map_df, lat="Lat", lon="Lon", size=pollutant, hover_name="City", color=pollutant,
                                  zoom=3, height=300, color_continuous_scale=px.colors.sequential.Viridis)
            fig.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Could not display map - missing or invalid coordinate data")
        
        st.markdown(f"### 📈 {pollutant} Over Time")
        try:
            fig, ax = plt.subplots(figsize=(10, 4))
            for city in selected_cities:
                city_data = filtered_df[filtered_df[city_col] == city]
                if 'Date' in city_data.columns:
                    series = city_data.groupby('Date')[actual_col].mean()
                    ax.plot(series.index, series.values, label=city)
            ax.set_xlabel("Date")
            ax.set_ylabel(f"{pollutant} (µg/m³)")
            ax.legend()
            st.pyplot(fig)
        except:
            st.info("Could not generate time series plot - missing or invalid date data")
        
        if 'PM2.5' in available_pollutants:
            st.markdown("### ⚠️ Health Alerts (Based on PM2.5)")
            pm25_col = [col for col in available_cols if col.lower() in ['pm25', 'pm2.5', 'pm2_5']][0]
            for city in selected_cities:
                try:
                    avg = filtered_df[filtered_df[city_col] == city][pm25_col].mean()
                    status, msg, css = interpret_pm25(avg)
                    st.markdown(f"<div class='health-alert {css}'><b>{city}: {status}</b> – {msg} (Avg PM2.5: {avg:.1f})</div>", unsafe_allow_html=True)
                except:
                    pass

elif page == "Prediction":
    available_cols = current_df.columns.tolist()
    
    # Check for PM2.5 with flexible column names
    pm25_cols = [col for col in available_cols if col.lower() in ['pm25', 'pm2.5', 'pm2_5']]
    
    if not pm25_cols:
        st.info("PM2.5 prediction is not available with the current dataset.")
    else:
        st.markdown("## Predict PM2.5")
        model_name = st.selectbox("Model", ["Linear Regression", "Random Forest", "Decision Tree", "Neural Network"])
        
        # Create sliders for available features only
        slider_values = {}
        
        # Flexible feature detection
        if any(col.lower() in ['pm10'] for col in available_cols):
            pm10_col = [col for col in available_cols if col.lower() == 'pm10'][0]
            slider_values[pm10_col] = st.slider("PM10", 0.0, 200.0, float(current_df[pm10_col].mean()))
        
        if any(col.lower() in ['no2', 'nitrogen dioxide'] for col in available_cols):
            no2_col = [col for col in available_cols if col.lower() in ['no2', 'nitrogen dioxide']][0]
            slider_values[no2_col] = st.slider("NO2", 0.0, 100.0, float(current_df[no2_col].mean()))
        
        if any(col.lower() in ['co', 'carbon monoxide', 'c0'] for col in available_cols):
            co_col = [col for col in available_cols if col.lower() in ['co', 'carbon monoxide', 'c0']][0]
            slider_values[co_col] = st.slider("CO", 0.0, 10.0, float(current_df[co_col].mean()))
        
        if any(col.lower() in ['o3', 'ozone'] for col in available_cols):
            o3_col = [col for col in available_cols if col.lower() in ['o3', 'ozone']][0]
            slider_values[o3_col] = st.slider("O3", 0.0, 200.0, float(current_df[o3_col].mean()))
        
        if any(col.lower() in ['temperature', 'temp'] for col in available_cols):
            temp_col = [col for col in available_cols if col.lower() in ['temperature', 'temp']][0]
            slider_values[temp_col] = st.slider("Temperature", -10.0, 40.0, float(current_df[temp_col].mean()))
        
        if any(col.lower() in ['humidity'] for col in available_cols):
            hum_col = [col for col in available_cols if col.lower() == 'humidity'][0]
            slider_values[hum_col] = st.slider("Humidity", 0.0, 100.0, float(current_df[hum_col].mean()))
        
        if any(col.lower() in ['wind speed', 'windspeed'] for col in available_cols):
            wind_col = [col for col in available_cols if col.lower() in ['wind speed', 'windspeed']][0]
            slider_values[wind_col] = st.slider("Wind Speed", 0.0, 20.0, float(current_df[wind_col].mean()))
        
        if any(col.lower() in ['city_mean_pm25', 'city_pm25'] for col in available_cols):
            city_pm_col = [col for col in available_cols if col.lower() in ['city_mean_pm25', 'city_pm25']][0]
            slider_values[city_pm_col] = st.slider("City Avg PM2.5", 0.0, 150.0, float(current_df[city_pm_col].mean()))
        
        features = slider_values.copy()
        
        # Add derived features if possible
        if any(col.lower() in ['pm10'] for col in available_cols) and len(pm25_cols) > 0:
            pm10_col = [col for col in available_cols if col.lower() == 'pm10'][0]
            features['PM_Ratio'] = features.get(pm10_col, 0) / (features.get(pm10_col, 0) * 0.5 + 1e-5)
        
        if any(col.lower() in ['humidity'] for col in available_cols) and any(col.lower() in ['temperature', 'temp'] for col in available_cols):
            hum_col = [col for col in available_cols if col.lower() == 'humidity'][0]
            temp_col = [col for col in available_cols if col.lower() in ['temperature', 'temp']][0]
            features['Humidity_Temp'] = features.get(hum_col, 0) * features.get(temp_col, 0)
        
        if any(col.lower() in ['o3', 'ozone'] for col in available_cols) and any(col.lower() in ['no2', 'nitrogen dioxide'] for col in available_cols):
            o3_col = [col for col in available_cols if col.lower() in ['o3', 'ozone']][0]
            no2_col = [col for col in available_cols if col.lower() in ['no2', 'nitrogen dioxide']][0]
            features['O3_NO2'] = features.get(o3_col, 0) / (features.get(no2_col, 0) + 1e-5)
        
        # Add temporal features
        features.update({
            'Month': 6, 'Day': 15, 'Weekday': 2, 'Is_Weekend': 0,
            'Lag_PM2.5': features.get('City_Mean_PM25', 0) * 0.9,
            'Rolling_PM2.5': features.get('City_Mean_PM25', 0)
        })
        
        input_df = pd.DataFrame([features])
        model_features = [col for col in input_df.columns if col in available_cols or col in ['PM_Ratio', 'Humidity_Temp', 'O3_NO2']]
        
        if len(model_features) > 0:
            model_map = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Neural Network": MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, early_stopping=True, random_state=42)
            }
            
            try:
                pipeline = Pipeline([("scaler", StandardScaler()), ("model", model_map[model_name])])
                pipeline.fit(current_df[model_features], current_df[pm25_cols[0]])
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
    st.markdown("## Policy Simulation")
    
    # Flexible pollutant detection for simulation
    sim_pollutants = {
        'PM10': ['pm10'],
        'NO2': ['no2', 'nitrogen dioxide'],
        'CO': ['co', 'carbon monoxide', 'c0'],
        'O3': ['o3', 'ozone']
    }
    
    available_sim_cols = []
    for standard_name, variants in sim_pollutants.items():
        for variant in variants:
            if variant in [col.lower() for col in current_df.columns]:
                available_sim_cols.append((standard_name, [col for col in current_df.columns if col.lower() == variant][0]))
                break
    
    if len(available_sim_cols) == 0:
        st.info("Policy simulation is not available with the current dataset.")
    else:
        col1, col2 = st.columns(2)
        base_values = {}
        
        with col1:
            for name, col in available_sim_cols[:2]:
                base_values[col] = st.number_input(name, value=float(current_df[col].mean()))
        
        with col2:
            for name, col in available_sim_cols[2:]:
                base_values[col] = st.number_input(name, value=float(current_df[col].mean()))
        
        st.markdown("### Adjustment (%)")
        adjustments = {}
        
        for name, col in available_sim_cols:
            adjustments[col] = st.slider(f"{name} Change", -50, 50, 0)
        
        if st.button("Simulate Impact"):
            new_values = simulate_policy_change(base_values, adjustments)
            
            try:
                # Check if we have PM2.5 data for target
                pm25_cols = [col for col in current_df.columns if col.lower() in ['pm25', 'pm2.5', 'pm2_5']]
                if not pm25_cols:
                    st.info("PM2.5 data not available for simulation target")
                    return
                
                model = RandomForestRegressor(random_state=42)
                model.fit(current_df[[col for name, col in available_sim_cols]], current_df[pm25_cols[0]])
                
                baseline_input = [base_values.get(col, 0) for name, col in available_sim_cols]
                new_input = [new_values.get(col, 0) for name, col in available_sim_cols]
                
                baseline = model.predict([baseline_input])[0]
                new_pred = model.predict([new_input])[0]
                
                st.metric("Baseline PM2.5", f"{baseline:.1f}")
                st.metric("Simulated PM2.5", f"{new_pred:.1f}", delta=f"{new_pred - baseline:.1f}")
                if new_pred < baseline:
                    st.success(f"✅ Policy reduces PM2.5 by {(baseline - new_pred):.1f}")
                else:
                    st.warning(f"⚠️ Policy increases PM2.5 by {(new_pred - baseline):.1f}")
            except Exception as e:
                st.info(f"Simulation couldn't be completed: {str(e)}")
