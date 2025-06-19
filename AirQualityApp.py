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
        <h1 style='text-align: center; color: #FF4B4B;'>🌍 Welcome to the Air Quality Dashboard</h1>
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
    st.markdown(
        """
        <h1 style='text-align: center; color: #FF4B4B;'>🌍 Air Quality Dashboard</h1>
        <h4 style='text-align: center; color: gray;'>Track & Visualize Global Pollutant Levels</h4>
        <br>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🌐 Filter by Location")
    country = st.selectbox("Select Country", sorted(df['Country'].unique()))
    filtered_df = df[df['Country'] == country]
    cities = st.multiselect("Select Cities", sorted(filtered_df['City'].unique()), default=sorted(filtered_df['City'].unique())[:1])
    filtered_df = filtered_df[filtered_df['City'].isin(cities)]

    pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    selected_pollutant = st.selectbox("Select Pollutant to Visualize", pollutants)

    st.markdown(f"### 📊 {selected_pollutant} Over Time")
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

    st.markdown(f"### 📊 Distribution of {selected_pollutant}")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(filtered_df, x=selected_pollutant, hue='City', kde=True, ax=ax, bins=30)
    st.pyplot(fig)

    st.markdown("### 📄 Average Pollutant Levels by City")
    avg_pollutants = filtered_df.groupby('City')[pollutants].mean().round(2)
    st.dataframe(avg_pollutants)

elif page == "Prediction":
    st.markdown("""
        <h1 style='text-align: center; color: #FF4B4B;'>🔮 Batch Predict PM2.5 Levels</h1>
        <h4 style='text-align: center; color: gray;'>Upload a CSV file to predict PM2.5 for multiple rows</h4>
        <br>
        """, unsafe_allow_html=True)

    sample_df = df[['PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed', 'PM2.5']].head()
    st.download_button("📥 Download Sample Format", sample_df.to_csv(index=False), "sample_input.csv")

    uploaded_file = st.file_uploader("Upload CSV File", type="csv")

    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        try:
            features = ['PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed']
            X_user = user_df[features]

            model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            X_train = df[features]
            y_train = df['PM2.5']
            model.fit(X_train, y_train)

            user_df['Predicted PM2.5'] = model.predict(X_user)
            st.success("✅ Prediction complete!")
            st.dataframe(user_df)

            st.download_button("📤 Download Predictions", user_df.to_csv(index=False), "predictions.csv")
        except Exception as e:
            st.error(f"⚠️ Error in input file: {e}")
