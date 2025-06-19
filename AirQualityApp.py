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
from sklearn.inspection import permutation_importance
from datetime import timedelta

# Set page config
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")

# Custom light theme background
st.markdown("""
    <style>
    body {
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

# Page selector
page = st.selectbox("Select Page", ["Home", "Dashboard", "Prediction"])

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv("AirQuality_Global.csv")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.drop_duplicates()
    df = df[(df.select_dtypes(include='number') >= 0).all(axis=1)]
    return df

df = load_data()

# Feature engineering
city_means = df.groupby('City')['PM2.5'].mean().to_dict()
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.weekday
df['Is_Weekend'] = df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)
df['City_Mean_PM25'] = df['City'].map(city_means)
df['PM_Ratio'] = df['PM10'] / (df['PM2.5'] + 1e-5)
df['Humidity_Temp'] = df['Humidity'] * df['Temperature']
df['O3_NO2'] = df['O3'] / (df['NO2'] + 1e-5)
df = df.sort_values(by=['City', 'Date'])
df['Lag_PM2.5'] = df.groupby('City')['PM2.5'].shift(1)
df['Rolling_PM2.5'] = df.groupby('City')['PM2.5'].transform(lambda x: x.rolling(3).mean())
df = df.dropna()

if page == "Home":
    st.markdown("""
        <h1 style='text-align: center; color: #FF4B4B;'>🌍 Welcome to the Air Quality Dashboard</h1>
        <h4 style='text-align: center; color: gray;'>Use the page selector above to explore air quality data by country and city</h4>
        <br>
        <ul>
            <li>Filter data by country and city</li>
            <li>Visualize pollutant trends and distributions</li>
            <li>Download filtered datasets for analysis</li>
            <li>Use prediction tool to estimate PM2.5 levels</li>
        </ul>
        <p><strong>WHO Guideline:</strong> 24h PM2.5 should not exceed <strong>15 µg/m³</strong></p>
    """, unsafe_allow_html=True)

elif page == "Dashboard":
    ...  # unchanged content for brevity

elif page == "Prediction":
    ...  # unchanged feature setup and input

    @st.cache_data
    def train_model(model):
        model.fit(X, y)
        return model

    model = train_model(model)

    ...  # unchanged prediction code

    # Optional Forecast Feature
    st.markdown("### ⏰ Optional: Forecast PM2.5 for the Next 7 Days")
    if st.button("Forecast Next Week"):
        future_dates = pd.date_range(df['Date'].max() + timedelta(days=1), periods=7)
        last_row = df.iloc[-1].copy()
        forecast_results = []

        for date in future_dates:
            features = last_row[feature_cols].copy()
            features['Month'] = date.month
            features['Day'] = date.day
            features['Weekday'] = date.weekday()
            features['Is_Weekend'] = 1 if date.weekday() >= 5 else 0
            prediction = model.predict([features])[0]
            forecast_results.append((date.strftime('%Y-%m-%d'), prediction))
            last_row['Lag_PM2.5'] = prediction
            last_row['Rolling_PM2.5'] = (last_row['Rolling_PM2.5'] + prediction) / 2  # simplistic

        forecast_df = pd.DataFrame(forecast_results, columns=["Date", "Predicted PM2.5"])
        st.dataframe(forecast_df)
        fig, ax = plt.subplots()
        sns.lineplot(data=forecast_df, x="Date", y="Predicted PM2.5", marker='o')
        ax.set_title("Forecasted PM2.5 for the Next 7 Days")
        ax.set_ylabel("PM2.5 µg/m³")
        st.pyplot(fig)

    # Feature Importance (only for Random Forest)
    if model_choice == "Random Forest":
        st.markdown("### 📊 Feature Importance (Random Forest)")
        importances = model.named_steps['model'].feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        st.dataframe(importance_df)
        fig, ax = plt.subplots()
        sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
        ax.set_title("Feature Importance")
        st.pyplot(fig)

    # Model comparison placeholder (if needed later)
    # (Can add more here for expanded evaluation)
