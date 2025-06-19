# app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Set page config at the top
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")

# Set light theme background color using custom CSS
st.markdown(
    """
    <style>
    body {
        background-color: #f8f9fa;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main page selector (not in sidebar)
page = st.selectbox("Select Page", ["Home", "Dashboard", "Prediction"])

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("AirQuality_Global.csv")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.drop_duplicates()
    df = df[(df.select_dtypes(include='number') >= 0).all(axis=1)]
    return df

df = load_data()

if page == "Home":
    st.markdown(
        """
        <h1 style='text-align: center; color: #FF4B4B;'> Welcome to the Air Quality Dashboard</h1>
        <h4 style='text-align: center; color: gray;'>Use the page selector above to explore air quality data by country and city</h4>
        <br>
        <ul>
            <li>Filter data by country and city</li>
            <li>Visualize pollutant trends and distributions</li>
            <li>Download filtered datasets for analysis</li>
            <li>Use prediction tool to estimate PM2.5 levels</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

elif page == "Dashboard":
    # Header
    st.markdown(
        """
        <h1 style='text-align: center; color: #FF4B4B;'>🌍 Air Quality Dashboard</h1>
        <h4 style='text-align: center; color: gray;'>Track & Visualize Global Pollutant Levels</h4>
        <br>
        """,
        unsafe_allow_html=True
    )

    # Filters
    st.markdown("### Filter by Location")
    country = st.selectbox("Select Country", sorted(df['Country'].unique()))
    filtered_df = df[df['Country'] == country]
    cities = st.multiselect("Select Cities", sorted(filtered_df['City'].unique()), default=sorted(filtered_df['City'].unique())[:1])
    filtered_df = filtered_df[filtered_df['City'].isin(cities)]

    # Pollutant Selection
    pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    selected_pollutant = st.selectbox("Select Pollutant to Visualize", pollutants)

    # Line Chart
    st.markdown(f"### {selected_pollutant} Over Time")
    fig, ax = plt.subplots(figsize=(12, 5))
    for city in cities:
        city_data = filtered_df[filtered_df['City'] == city]
        city_avg = city_data.groupby('Date')[selected_pollutant].mean()
        ax.plot(city_avg.index, city_avg.values, label=city)
    ax.set_ylabel(f"{selected_pollutant} concentration")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Distribution
    st.markdown(f"### Distribution of {selected_pollutant}")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(filtered_df, x=selected_pollutant, hue='City', kde=True, ax=ax, bins=30)
    st.pyplot(fig)

    # Data Table
    st.markdown("### Average Pollutant Levels by City")
    avg_pollutants = filtered_df.groupby('City')[pollutants].mean().round(2)
    st.dataframe(avg_pollutants)

elif page == "Prediction":
    st.markdown("""
        <h1 style='text-align: center; color: #FF4B4B;'>🔮 Predict PM2.5 Levels</h1>
        <h4 style='text-align: center; color: gray;'>Enter pollutant & weather data to estimate PM2.5 concentration</h4>
        <br>
        """, unsafe_allow_html=True)

    # Input features
    st.markdown("### Enter Features")
    PM10 = st.number_input("PM10", 0.0, 500.0, 50.0)
    NO2 = st.number_input("NO2", 0.0, 200.0, 20.0)
    SO2 = st.number_input("SO2", 0.0, 100.0, 10.0)
    CO = st.number_input("CO", 0.0, 15.0, 1.0)
    O3 = st.number_input("O3", 0.0, 300.0, 50.0)
    Temperature = st.number_input("Temperature (°C)", -30.0, 50.0, 25.0)
    Humidity = st.number_input("Humidity (%)", 0.0, 100.0, 50.0)
    WindSpeed = st.number_input("Wind Speed (km/h)", 0.0, 100.0, 10.0)

    input_data = np.array([[PM10, NO2, SO2, CO, O3, Temperature, Humidity, WindSpeed]])

    # Feature columns
    features = ['PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed']
    X = df[features]
    y = df['PM2.5']

    # Model selection
    st.markdown("### Choose a Model")
    model_choice = st.selectbox("Select Model", ["Linear Regression", "Random Forest", "Decision Tree", "Neural Network"])

    if model_choice == "Linear Regression":
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
    elif model_choice == "Random Forest":
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
    elif model_choice == "Decision Tree":
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', DecisionTreeRegressor(random_state=42))
        ])
    elif model_choice == "Neural Network":
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, early_stopping=True, random_state=42))
        ])

    # Train the model
    model.fit(X, y)

    # Make prediction
    predicted_pm25 = model.predict(input_data)[0]
    st.success(f"Predicted PM2.5 Level using {model_choice}: {predicted_pm25:.2f} µg/m³")
