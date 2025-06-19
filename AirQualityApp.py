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
        """, unsafe_allow_html=True)

elif page == "Dashboard":
    # [Dashboard code remains unchanged]
    ...

elif page == "Prediction":
    st.markdown("""
        <h1 style='text-align: center; color: #FF4B4B;'>🔮 Predict PM2.5 Levels</h1>
        <h4 style='text-align: center; color: gray;'>Estimate PM2.5 concentration using multiple features</h4>
        <br>
        """, unsafe_allow_html=True)

    feature_cols = [
        'PM10', 'NO2', 'SO2', 'CO', 'O3',
        'Temperature', 'Humidity', 'Wind Speed',
        'Month', 'Day', 'Weekday', 'Is_Weekend',
        'City_Mean_PM25', 'PM_Ratio', 'Humidity_Temp', 'O3_NO2',
        'Lag_PM2.5', 'Rolling_PM2.5'
    ]
    X = df[feature_cols]
    y = df['PM2.5']

    model_choice = st.selectbox("Choose Model", ["Linear Regression", "Random Forest", "Decision Tree", "Neural Network"])
    if model_choice == "Linear Regression":
        model = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
    elif model_choice == "Random Forest":
        model = Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor(n_estimators=100, random_state=42))])
    elif model_choice == "Decision Tree":
        model = Pipeline([('scaler', StandardScaler()), ('model', DecisionTreeRegressor(random_state=42))])
    elif model_choice == "Neural Network":
        model = Pipeline([('scaler', StandardScaler()), ('model', MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, early_stopping=True, random_state=42))])

    model.fit(X, y)

    st.markdown("### 📥 Enter Feature Values")
    input_dict = {col: st.number_input(col, value=float(df[col].mean())) for col in feature_cols}
    input_array = np.array([list(input_dict.values())])
    predicted = model.predict(input_array)[0]
    st.success(f"🌫️ Predicted PM2.5 Level using {model_choice}: {predicted:.2f} µg/m³")

    st.markdown("### 📤 Or Upload a CSV File for Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file with the same feature columns", type=["csv"])

    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            user_X = user_df[feature_cols]
            user_df['Predicted_PM2.5'] = model.predict(user_X)
            st.success("✅ Batch prediction complete!")
            st.dataframe(user_df)
            st.download_button("Download Predictions", user_df.to_csv(index=False), file_name="predicted_pm25.csv", mime="text/csv")
        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")
