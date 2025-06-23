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

# Page configuration
st.set_page_config(page_title="Air Quality Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS styling
st.markdown("""
    <style>
    body { background-color: #f8f9fa; font-family: Arial, sans-serif; }
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

# Load data
@st.cache_resource
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("AirQuality_Final_Processed.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data(uploaded_file)

# Dataset info
with st.sidebar.expander("ℹ️ Dataset Info"):
    if uploaded_file is not None:
        st.success("✅ Using uploaded dataset.")
    else:
        st.info("📁 Using default dataset.")
    st.write(f"Records: {df.shape[0]} | Columns: {df.shape[1]}")

# AQI calculator
def calculate_aqi(pm25):
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

# PM2.5 interpretation
def interpret_pm25(value):
    if value <= 12:
        return ("🟢 Good", "Air quality is satisfactory.", "good")
    elif value <= 35.4:
        return ("🟡 Moderate", "Acceptable for most, but sensitive groups may be affected.", "moderate")
    elif value <= 55.4:
        return ("🟠 Unhealthy for Sensitive Groups", "People with heart/lung disease, children, and older adults should reduce prolonged exertion.", "unhealthy-sensitive")
    elif value <= 150.4:
        return ("🔴 Unhealthy", "Everyone may experience health effects.", "unhealthy")
    else:
        return ("⚫ Very Unhealthy", "Health warnings for everyone; avoid outdoor activities.", "very-unhealthy")

# Policy simulation logic
def simulate_policy_change(base_values, adjustments):
    adjusted = base_values.copy()
    for k, v in adjustments.items():
        adjusted[k] *= (1 + v / 100)
    return adjusted

# Sidebar page selection
page = st.sidebar.selectbox("Select Page", ["Home", "Dashboard", "Prediction", "Policy Simulation"])

# Home Page
if page == "Home":
    st.markdown("""
        <h1 style='text-align: center; color: #FF4B4B;'>Welcome to the Air Quality Dashboard</h1>
        <h4 style='text-align: center; color: gray;'>Track, analyze, and predict air quality for healthier communities</h4>
        <br>
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
            <h3>📌 Key Features:</h3>
            <ul>
                <li><b>🌐 Dashboard:</b> Visualize air quality trends by city and pollutant</li>
                <li><b>🔮 Prediction:</b> Estimate PM2.5 levels using machine learning</li>
                <li><b>📜 Policy Simulation:</b> Test how emission reductions affect air quality</li>
            </ul>
        </div>
        <br>
        <h3>💡 Did You Know?</h3>
        <blockquote>
            PM2.5 particles are 30x smaller than a human hair and can enter your bloodstream.
        </blockquote>
    """, unsafe_allow_html=True)

# Dashboard Page
elif page == "Dashboard":
    st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>Air Quality Dashboard</h1>", unsafe_allow_html=True)

    country = st.sidebar.selectbox("Select Country", sorted(df['Country'].unique()))
    filtered_df = df[df['Country'] == country]

    cities = st.sidebar.multiselect("Select Cities", sorted(filtered_df['City'].unique()), default=sorted(filtered_df['City'].unique())[:1])
    pollutant = st.sidebar.selectbox("Select Pollutant", ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3'])

    filtered_df = filtered_df[filtered_df['City'].isin(cities)]

    st.markdown("### 📍 City Locations")
    city_coords = {
        'Bangkok': (13.7563, 100.5018),
        'Paris': (48.8566, 2.3522),
        # Add more cities as needed
    }
    map_df = pd.DataFrame({
        'City': cities,
        'Lat': [city_coords.get(city, (0, 0))[0] for city in cities],
        'Lon': [city_coords.get(city, (0, 0))[1] for city in cities],
        'PM2.5': [filtered_df[filtered_df['City'] == city]['PM2.5'].mean() for city in cities]
    })
    fig = px.scatter_mapbox(map_df, lat="Lat", lon="Lon", hover_name="City", size="PM2.5", color="PM2.5",
                            zoom=4, height=300, color_continuous_scale=px.colors.sequential.Viridis)
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"### 📈 {pollutant} Over Time")
    fig, ax = plt.subplots(figsize=(12, 5))
    for city in cities:
        city_data = filtered_df[filtered_df['City'] == city]
        city_avg = city_data.groupby('Date')[pollutant].mean()
        ax.plot(city_avg.index, city_avg.values, label=city)
    ax.set_ylabel(f"{pollutant} (µg/m³)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.markdown("### ⚠️ Health Alerts")
    for city in cities:
        avg_pm25 = filtered_df[filtered_df['City'] == city]['PM2.5'].mean()
        status, message, css_class = interpret_pm25(avg_pm25)
        st.markdown(f"""
            <div class="health-alert {css_class}">
                <b>{city}: {status}</b> – {message} (Avg PM2.5: {avg_pm25:.1f} µg/m³)
            </div>
        """, unsafe_allow_html=True)

    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button("Download Filtered CSV", data=csv, file_name=f"air_quality_{country}.csv", mime="text/csv")

# Prediction Page
elif page == "Prediction":
    st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>Predict PM2.5 Levels</h1>", unsafe_allow_html=True)

    model_name = st.selectbox("Choose Model", ["Linear Regression", "Random Forest", "Decision Tree", "Neural Network"])

    st.markdown("### Enter Environmental Conditions")
    col1, col2 = st.columns(2)
    with col1:
        pm10 = st.slider("PM10", 0.0, 200.0, float(df['PM10'].mean()))
        no2 = st.slider("NO2", 0.0, 100.0, float(df['NO2'].mean()))
        temp = st.slider("Temperature", -10.0, 40.0, float(df['Temperature'].mean()))
        humidity = st.slider("Humidity", 0.0, 100.0, float(df['Humidity'].mean()))
    with col2:
        co = st.slider("CO", 0.0, 10.0, float(df['CO'].mean()))
        o3 = st.slider("O3", 0.0, 200.0, float(df['O3'].mean()))
        wind = st.slider("Wind Speed", 0.0, 20.0, float(df['Wind Speed'].mean()))
        city_avg = st.slider("City Mean PM2.5", 0.0, 150.0, float(df['City_Mean_PM25'].mean()))

    input_data = pd.DataFrame([{
        'PM10': pm10, 'NO2': no2, 'SO2': df['SO2'].mean(), 'CO': co, 'O3': o3,
        'Temperature': temp, 'Humidity': humidity, 'Wind Speed': wind,
        'City_Mean_PM25': city_avg,
        'PM_Ratio': pm10 / (pm10 * 0.5 + 1e-5),
        'Humidity_Temp': humidity * temp,
        'O3_NO2': o3 / (no2 + 1e-5),
        'Month': 6, 'Day': 15, 'Weekday': 2, 'Is_Weekend': 0,
        'Lag_PM2.5': city_avg * 0.9,
        'Rolling_PM2.5': city_avg
    }])

    X = df[input_data.columns]
    y = df['PM2.5']

    if model_name == "Linear Regression":
        model = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
    elif model_name == "Random Forest":
        model = Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor(n_estimators=100, random_state=42))])
    elif model_name == "Decision Tree":
        model = Pipeline([('scaler', StandardScaler()), ('model', DecisionTreeRegressor(random_state=42))])
    elif model_name == "Neural Network":
        model = Pipeline([('scaler', StandardScaler()), ('model', MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, early_stopping=True, random_state=42))])

    model.fit(X, y)
    pred = model.predict(input_data)[0]
    aqi = calculate_aqi(pred)
    status, msg, css_class = interpret_pm25(pred)

    st.metric("Predicted PM2.5", f"{pred:.1f} µg/m³")
    st.metric("AQI", f"{aqi:.0f}")
    st.markdown(f"<div class='health-alert {css_class}'><b>Health Impact: {status}</b> – {msg}</div>", unsafe_allow_html=True)

# Policy Simulation Page
elif page == "Policy Simulation":
    st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🏛️ Policy Simulation</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        pm10_base = st.number_input("PM10", value=float(df['PM10'].mean()))
        no2_base = st.number_input("NO2", value=float(df['NO2'].mean()))
    with col2:
        co_base = st.number_input("CO", value=float(df['CO'].mean()))
        o3_base = st.number_input("O3", value=float(df['O3'].mean()))

    st.markdown("### ⚙️ Policy Adjustment (%)")
    pm10_change = st.slider("PM10 Change", -50, 50, 0)
    no2_change = st.slider("NO2 Change", -50, 50, 0)
    co_change = st.slider("CO Change", -50, 50, 0)
    o3_change = st.slider("O3 Change", -50, 50, 0)

    if st.button("Simulate Policy Impact"):
        adjustments = {'PM10': -pm10_change, 'NO2': -no2_change, 'CO': -co_change, 'O3': -o3_change}
        base_vals = {'PM10': pm10_base, 'NO2': no2_base, 'CO': co_base, 'O3': o3_base}
        new_vals = simulate_policy_change(base_vals, adjustments)

        model = RandomForestRegressor(random_state=42)
        model.fit(df[['PM10', 'NO2', 'CO', 'O3']], df['PM2.5'])

        baseline = model.predict([[pm10_base, no2_base, co_base, o3_base]])[0]
        new_pred = model.predict([[new_vals['PM10'], new_vals['NO2'], new_vals['CO'], new_vals['O3']]])[0]

        st.metric("Baseline PM2.5", f"{baseline:.1f} µg/m³")
        st.metric("Simulated PM2.5", f"{new_pred:.1f} µg/m³", delta=f"{new_pred - baseline:.1f} µg/m³")
        if new_pred < baseline:
            st.success(f"✅ Policy reduces PM2.5 by {(baseline - new_pred):.1f} µg/m³")
        else:
            st.warning(f"⚠️ Policy increases PM2.5 by {(new_pred - baseline):.1f} µg/m³")
