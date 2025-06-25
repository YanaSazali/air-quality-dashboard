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
    </style>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])

@st.cache_resource
def load_data(uploaded_file=None):
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file else pd.read_csv("AirQuality_Final_Processed.csv")
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.dropna(subset=['Date'], inplace=True)
        else:
            df['Date'] = pd.NaT
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    return df

df = load_data(uploaded_file)

with st.sidebar.expander("ℹ️ Dataset Info"):
    st.success("✅ Uploaded dataset used." if uploaded_file else "📁 Default dataset used.")
    st.write(f"Records: {df.shape[0]} | Columns: {df.shape[1]}")

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
            <h3>📌 Key Features:</h3>
            <ul>
                <li><b>🌐 Dashboard:</b> Visualize air quality trends</li>
                <li><b>🔮 Prediction:</b> Estimate PM2.5 using ML models</li>
                <li><b>📜 Policy Simulation:</b> Assess pollution reduction impact</li>
            </ul>
        </div>
        <h3>💡 Did You Know?</h3>
        <blockquote>
            PM2.5 particles are smaller than a human hair and can enter your lungs.
        </blockquote>
    """, unsafe_allow_html=True)

elif page == "Dashboard":
    st.markdown("## 🌍 Air Quality Dashboard")
    if not all(col in df.columns for col in ['Country', 'City']):
        st.warning("Dataset lacks essential columns: 'Country', 'City'.")
    else:
        country = st.sidebar.selectbox("Select Country", sorted(df['Country'].dropna().unique()))
        cities = st.sidebar.multiselect("Select Cities", sorted(df[df['Country'] == country]['City'].dropna().unique()))
        pollutant_choices = [col for col in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3'] if col in df.columns]
        if not pollutant_choices:
            st.warning("No pollutant columns available.")
        else:
            pollutant = st.sidebar.selectbox("Select Pollutant", pollutant_choices)
            filtered_df = df[(df['Country'] == country) & (df['City'].isin(cities))]

            st.markdown("### 📍 City Locations (if available)")
            city_coords = {'Bangkok': (13.75, 100.5), 'Paris': (48.85, 2.35)}
            map_df = pd.DataFrame({
                'City': cities,
                'Lat': [city_coords.get(city, (0, 0))[0] for city in cities],
                'Lon': [city_coords.get(city, (0, 0))[1] for city in cities],
                pollutant: [filtered_df[filtered_df['City'] == city][pollutant].mean() for city in cities]
            })
            fig = px.scatter_mapbox(map_df, lat="Lat", lon="Lon", size=pollutant, hover_name="City", color=pollutant,
                                    zoom=3, height=300, color_continuous_scale=px.colors.sequential.Viridis)
            fig.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"### 📈 {pollutant} Over Time")
            if pollutant not in filtered_df.columns or filtered_df[pollutant].dropna().empty:
                st.warning(f"No data for {pollutant}.")
            else:
                fig, ax = plt.subplots(figsize=(10, 4))
                for city in cities:
                    series = filtered_df[filtered_df['City'] == city].groupby('Date')[pollutant].mean()
                    ax.plot(series.index, series.values, label=city)
                ax.set_xlabel("Date")
                ax.set_ylabel(f"{pollutant} (µg/m³)")
                ax.legend()
                st.pyplot(fig)

            if 'PM2.5' in df.columns:
                st.markdown("### ⚠️ Health Alerts (Based on PM2.5)")
                for city in cities:
                    avg = filtered_df[filtered_df['City'] == city]['PM2.5'].mean()
                    status, msg, css = interpret_pm25(avg)
                    st.markdown(f"<div class='health-alert {css}'><b>{city}: {status}</b> – {msg} (Avg PM2.5: {avg:.1f})</div>", unsafe_allow_html=True)
            else:
                st.info("Health alerts are disabled because PM2.5 is not available.")

elif page == "Prediction":
    if 'PM2.5' not in df.columns:
        st.warning("Prediction is unavailable because PM2.5 values are missing from the dataset.")
    else:
        st.markdown("## 🔮 Predict PM2.5")
        required_cols = ['PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed', 'City_Mean_PM25']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            st.warning(f"Cannot run predictions: Missing columns: {', '.join(missing)}")
        else:
            model_name = st.selectbox("Model", ["Linear Regression", "Random Forest", "Decision Tree", "Neural Network"])
            default_vals = {col: df[col].mean() for col in required_cols}
            pm10 = st.slider("PM10", 0.0, 200.0, float(default_vals['PM10']))
            no2 = st.slider("NO2", 0.0, 100.0, float(default_vals['NO2']))
            co = st.slider("CO", 0.0, 10.0, float(default_vals['CO']))
            o3 = st.slider("O3", 0.0, 200.0, float(default_vals['O3']))
            temp = st.slider("Temperature", -10.0, 40.0, float(default_vals['Temperature']))
            humidity = st.slider("Humidity", 0.0, 100.0, float(default_vals['Humidity']))
            wind = st.slider("Wind Speed", 0.0, 20.0, float(default_vals['Wind Speed']))
            city_avg = st.slider("City Avg PM2.5", 0.0, 150.0, float(default_vals['City_Mean_PM25']))

            features = {
                'PM10': pm10, 'NO2': no2, 'SO2': default_vals['SO2'], 'CO': co, 'O3': o3,
                'Temperature': temp, 'Humidity': humidity, 'Wind Speed': wind,
                'City_Mean_PM25': city_avg,
                'PM_Ratio': pm10 / (pm10 * 0.5 + 1e-5),
                'Humidity_Temp': humidity * temp,
                'O3_NO2': o3 / (no2 + 1e-5),
                'Month': 6, 'Day': 15, 'Weekday': 2, 'Is_Weekend': 0,
                'Lag_PM2.5': city_avg * 0.9,
                'Rolling_PM2.5': city_avg
            }

            input_df = pd.DataFrame([features])
            model_features = input_df.columns.tolist()

            model_map = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Neural Network": MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, early_stopping=True, random_state=42)
            }

            pipeline = Pipeline([("scaler", StandardScaler()), ("model", model_map[model_name])])
            pipeline.fit(df[model_features], df['PM2.5'])
            prediction = pipeline.predict(input_df)[0]
            aqi = calculate_aqi(prediction)
            status, msg, css = interpret_pm25(prediction)

            st.metric("Predicted PM2.5", f"{prediction:.1f} µg/m³")
            st.metric("AQI", f"{aqi:.0f}")
            st.markdown(f"<div class='health-alert {css}'><b>{status}</b> – {msg}</div>", unsafe_allow_html=True)

elif page == "Policy Simulation":
    st.markdown("## 🏫 Policy Simulation")
    sim_cols = ['PM10', 'NO2', 'CO', 'O3']
    if not all(col in df.columns for col in sim_cols):
        st.warning(f"Missing required columns for simulation: {', '.join([c for c in sim_cols if c not in df.columns])}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            pm10_base = st.number_input("PM10", value=float(df['PM10'].mean()))
            no2_base = st.number_input("NO2", value=float(df['NO2'].mean()))
        with col2:
            co_base = st.number_input("CO", value=float(df['CO'].mean()))
            o3_base = st.number_input("O3", value=float(df['O3'].mean()))

        st.markdown("### Adjustment (%)")
        pm10_chg = st.slider("PM10 Change", -50, 50, 0)
        no2_chg = st.slider("NO2 Change", -50, 50, 0)
        co_chg = st.slider("CO Change", -50, 50, 0)
        o3_chg = st.slider("O3 Change", -50, 50, 0)

        if st.button("Simulate Impact"):
            adjustments = {'PM10': -pm10_chg, 'NO2': -no2_chg, 'CO': -co_chg, 'O3': -o3_chg}
            base_vals = {'PM10': pm10_base, 'NO2': no2_base, 'CO': co_base, 'O3': o3_base}
            new_vals = simulate_policy_change(base_vals, adjustments)

            model = RandomForestRegressor(random_state=42)
            model.fit(df[sim_cols], df['PM2.5'])

            baseline = model.predict([[pm10_base, no2_base, co_base, o3_base]])[0]
            new_pred = model.predict([[new_vals['PM10'], new_vals['NO2'], new_vals['CO'], new_vals['O3']]])[0]

            st.metric("Baseline PM2.5", f"{baseline:.1f}")
            st.metric("Simulated PM2.5", f"{new_pred:.1f}", delta=f"{new_pred - baseline:.1f}")
            if new_pred < baseline:
                st.success(f"✅ Policy reduces PM2.5 by {(baseline - new_pred):.1f}")
            else:
                st.warning(f"⚠️ Policy increases PM2.5 by {(new_pred - baseline):.1f}")
