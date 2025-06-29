# NOTE: This is a modified version of your Streamlit app to ensure the Prediction and Policy Simulation
# pages remain usable even if the uploaded dataset lacks necessary features.

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

# Set Streamlit page config
st.set_page_config(page_title="Air Quality Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
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

@st.cache_data
def load_default_data():
    try:
        df = pd.read_csv("AirQuality_Final_Processed.csv")
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            df['Date'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
            df.dropna(subset=['Date'], inplace=True)
        return df
    except:
        return pd.DataFrame()

default_df = load_default_data()

uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])

@st.cache_data
def load_uploaded_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            df['Date'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
            df.dropna(subset=['Date'], inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading uploaded file: {e}")
        return None

def standardize_columns(df):
    mapping = {
        'pm25': 'PM2.5', 'pm2.5': 'PM2.5', 'pm10': 'PM10', 'no2': 'NO2',
        'so2': 'SO2', 'co': 'CO', 'o3': 'O3', 'c0': 'CO', 'temperature': 'Temperature',
        'humidity': 'Humidity', 'wind speed': 'Wind Speed', 'city': 'City', 'country': 'Country'
    }
    df.columns = df.columns.str.lower()
    df.rename(columns=mapping, inplace=True)
    return df

# Load and prepare the dataset
if uploaded_file:
    df = load_uploaded_data(uploaded_file)
    if df is not None:
        df = standardize_columns(df)
        current_df = df.copy()
        show_uploaded_data = True
    else:
        st.warning("⚠️ Uploaded file could not be used. Falling back to default dataset.")
        current_df = default_df.copy()
        show_uploaded_data = False
else:
    current_df = default_df.copy()
    show_uploaded_data = False

# Helper: Calculate AQI from PM2.5
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

# Helper: Interpret PM2.5 Level
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

# Helper: Simulate policy adjustments
def simulate_policy_change(base_values, adjustments):
    adjusted = base_values.copy()
    for k, v in adjustments.items():
        adjusted[k] = adjusted.get(k, 0) * (1 + v / 100)
    return adjusted

# Sidebar: page selector
page = st.sidebar.selectbox("Select Page", ["Home", "Prediction", "Policy Simulation"])

if page == "Home":
    st.markdown("""
        <h1 style='text-align: center; color: #FF4B4B;'>Welcome to the Air Quality Dashboard</h1>
        <h4 style='text-align: center; color: gray;'>Track, analyze, and predict air quality for healthier communities</h4>
        <br>
        <ul>
            <li><b>Prediction:</b> Estimate PM2.5 using ML models</li>
            <li><b>Policy Simulation:</b> Assess pollution reduction impact</li>
        </ul>
    """, unsafe_allow_html=True)

elif page == "Prediction":
    st.markdown("## Predict PM2.5")

    available_cols = current_df.columns.tolist()
    pm25_cols = [col for col in available_cols if col.lower() in ['pm25', 'pm2.5', 'pm2_5']]

    if not pm25_cols:
        st.warning("⚠️ PM2.5 not found. Falling back to default dataset.")
        current_df = default_df.copy()
        available_cols = current_df.columns.tolist()
        pm25_cols = [col for col in available_cols if col.lower() in ['pm25', 'pm2.5', 'pm2_5']]

    model_name = st.selectbox("Model", ["Linear Regression", "Random Forest", "Decision Tree", "Neural Network"])
    slider_values = {}

    for col in ['pm10', 'no2', 'co', 'o3', 'temperature', 'humidity', 'wind speed']:
        matches = [c for c in available_cols if c.lower() == col]
        if matches:
            col_name = matches[0]
            st_val = float(current_df[col_name].mean())
            slider_values[col_name] = st.slider(col_name.upper(), 0.0, float(st_val * 2), st_val)

    features = slider_values.copy()

    input_df = pd.DataFrame([features])
    model_features = list(input_df.columns)

    if model_features:
        model_map = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Neural Network": MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=42)
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
            st.error("Prediction failed: " + str(e))
    else:
        st.warning("Not enough input features available for prediction.")

elif page == "Policy Simulation":
    st.markdown("## Policy Simulation")
    pollutant_candidates = {
        'PM10': ['pm10'], 'NO2': ['no2'], 'CO': ['co', 'c0'], 'O3': ['o3']
    }

    available_sim_cols = []
    for name, variants in pollutant_candidates.items():
        for variant in variants:
            match = [col for col in current_df.columns if col.lower() == variant]
            if match:
                available_sim_cols.append((name, match[0]))
                break

    if not available_sim_cols:
        st.warning("⚠️ No pollutants found for simulation. Falling back to default dataset.")
        current_df = default_df.copy()
        # Retry detection
        available_sim_cols = []
        for name, variants in pollutant_candidates.items():
            for variant in variants:
                match = [col for col in current_df.columns if col.lower() == variant]
                if match:
                    available_sim_cols.append((name, match[0]))
                    break

    base_values = {}
    st.markdown("### Baseline Pollution Values")
    for name, col in available_sim_cols:
        base_values[col] = st.number_input(f"{name} (µg/m³)", value=float(current_df[col].mean()))

    st.markdown("### Adjustment (%)")
    adjustments = {col: st.slider(f"{name} Change (%)", -50, 50, 0) for name, col in available_sim_cols}

    pm25_cols = [col for col in current_df.columns if col.lower() in ['pm25', 'pm2.5', 'pm2_5']]
    if pm25_cols:
        if st.button("Simulate Impact"):
            try:
                model = RandomForestRegressor(random_state=42)
                input_cols = [col for name, col in available_sim_cols]
                model.fit(current_df[input_cols], current_df[pm25_cols[0]])
                
                baseline = model.predict([list(base_values.values())])[0]
                adjusted_vals = simulate_policy_change(base_values, adjustments)
                prediction = model.predict([list(adjusted_vals.values())])[0]

                st.metric("Baseline PM2.5", f"{baseline:.1f}")
                st.metric("Simulated PM2.5", f"{prediction:.1f}", delta=f"{prediction - baseline:.1f}")
            except Exception as e:
                st.error("Simulation error: " + str(e))
    else:
        st.warning("⚠️ PM2.5 data not found for prediction target.")
