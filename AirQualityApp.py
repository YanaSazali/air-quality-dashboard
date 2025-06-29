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

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])

@st.cache_resource
def load_data(uploaded_file=None):
    try:
        # Show loading status
        if uploaded_file:
            loading_placeholder = st.empty()
            loading_placeholder.markdown("""
                <div style="display: flex; align-items: center;">
                    <div class="loading-spinner"></div>
                    <span>Loading dataset...</span>
                </div>
            """, unsafe_allow_html=True)
        
        df = pd.read_csv(uploaded_file) if uploaded_file else pd.read_csv("AirQuality_Final_Processed.csv")
        
        # Handle date conversion gracefully
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            df['Date'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
            df.dropna(subset=['Date'], inplace=True)
        else:
            df['Date'] = pd.NaT
            
        # Clear loading status after processing
        if uploaded_file:
            loading_placeholder.empty()
            
        return df
    except Exception as e:
        if uploaded_file:
            loading_placeholder.empty()
        st.error(f"Error loading data: {e}")
        st.stop()

# Show initial loading state
if uploaded_file and 'df' not in st.session_state:
    st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <div class="loading-spinner"></div>
            <span>Processing uploaded dataset...</span>
        </div>
    """, unsafe_allow_html=True)
    st.session_state.df_loading = True

df = load_data(uploaded_file)

# Clear loading state after data is loaded
if 'df_loading' in st.session_state:
    st.session_state.df = df
    del st.session_state.df_loading
    st.rerun()

with st.sidebar.expander("ℹ️ Dataset Info"):
    st.success("✅ Uploaded dataset used." if uploaded_file else "📁 Default dataset used.")
    st.write(f"Records: {df.shape[0]} | Columns: {df.shape[1]}")
    if st.checkbox("Show column names"):
        st.write(df.columns.tolist())

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

    available_cols = df.columns.tolist()
    has_country = 'Country' in available_cols
    has_city = 'City' in available_cols

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

    pollutant_choices = [col for col in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3'] if col in available_cols]
    pollutant = st.sidebar.selectbox("Select Pollutant", pollutant_choices if pollutant_choices else ["No pollutants available"])

    if has_country and has_city and len(selected_cities) > 0 and pollutant_choices:
        filtered_df = df[(df['Country'] == country) & (df['City'].isin(selected_cities))]

        st.markdown("### 📍 City Locations (if coordinates available)")
        try:
            city_coords = {'Bangkok': (13.75, 100.5), 'Paris': (48.85, 2.35)}
            map_df = pd.DataFrame({
                'City': selected_cities,
                'Lat': [city_coords.get(city, (0, 0))[0] for city in selected_cities],
                'Lon': [city_coords.get(city, (0, 0))[1] for city in selected_cities],
                pollutant: [filtered_df[filtered_df['City'] == city][pollutant].mean() for city in selected_cities]
            })
            fig = px.scatter_mapbox(map_df, lat="Lat", lon="Lon", size=pollutant, hover_name="City", color=pollutant,
                                  zoom=3, height=300, color_continuous_scale=px.colors.sequential.Viridis)
            fig.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig, use_container_width=True)
        except:
            pass

    st.markdown(f"### 📈 {pollutant} Over Time")

    # Check for valid date and pollutant data
    valid_plot_data = False
    for city in selected_cities:
        city_data = filtered_df[filtered_df['City'] == city]
        if not city_data.empty and 'Date' in city_data.columns and city_data['Date'].notna().any() and city_data[pollutant].notna().any():
            valid_plot_data = True
            break

    if valid_plot_data:
        fig, ax = plt.subplots(figsize=(10, 4))
        for city in selected_cities:
            city_data = filtered_df[filtered_df['City'] == city].copy()
            city_data = city_data.dropna(subset=['Date', pollutant])
            if not city_data.empty:
                series = city_data.groupby('Date')[pollutant].mean()
                ax.plot(series.index, series.values, label=city)
        ax.set_xlabel("Date")
        ax.set_ylabel(f"{pollutant} (µg/m³)")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("⛔ Time series plot not available: Missing or invalid 'Date' or pollutant values.")

    if 'PM2.5' in available_cols:
        st.markdown("### ⚠️ Health Alerts (Based on PM2.5)")
        for city in selected_cities:
            try:
                avg = filtered_df[filtered_df['City'] == city]['PM2.5'].mean()
                status, msg, css = interpret_pm25(avg)
                st.markdown(
                    f"<div class='health-alert {css}'><b>{city}: {status}</b> – {msg} (Avg PM2.5: {avg:.1f})</div>",
                    unsafe_allow_html=True
                )
            except:
                pass


        st.markdown("### 🧾 Pollutant Data Table")
        display_columns = ['Date', 'City', 'Country'] + pollutant_choices
        available_table_cols = [col for col in display_columns if col in filtered_df.columns]
        if available_table_cols:
            st.dataframe(filtered_df[available_table_cols].sort_values("Date", ascending=False).reset_index(drop=True))
        else:
            st.warning("Some pollutant columns are missing from the dataset.")

elif page == "Prediction":
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
