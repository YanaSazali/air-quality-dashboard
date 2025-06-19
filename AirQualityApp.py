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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    """, unsafe_allow_html=True)

elif page == "Dashboard":
    st.markdown("""
        <h1 style='text-align: center; color: #FF4B4B;'>🌍 Air Quality Dashboard</h1>
        <h4 style='text-align: center; color: gray;'>Track & Visualize Global Pollutant Levels</h4>
        <br>
    """, unsafe_allow_html=True)

    st.markdown("### 🌐 Filter by Location")
    country = st.selectbox("Select Country", sorted(df['Country'].unique()))
    filtered_df = df[df['Country'] == country]
    cities = st.multiselect("Select Cities", sorted(filtered_df['City'].unique()), default=sorted(filtered_df['City'].unique())[:1])
    filtered_df = filtered_df[filtered_df['City'].isin(cities)]

    pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']

    with st.expander("ℹ️ What do these pollutants mean?"):
        st.markdown("""
        - **PM2.5**: Fine particles harmful to lungs.
        - **PM10**: Larger particles (e.g. dust).
        - **NO2**: Emitted from vehicles and factories.
        - **SO2**: Linked to coal and oil combustion.
        - **CO**: Carbon monoxide from incomplete combustion.
        - **O3**: Ozone; forms from reactions between pollutants.
        """)

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

    # Summary Insights
    st.markdown("### 🧐 Summary Insights")
    top_cities = (
        df[df['Country'] == country]
        .groupby('City')['PM2.5']
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )
    st.markdown("#### 🌆 Top 5 Most Polluted Cities (by PM2.5)")
    st.dataframe(top_cities)

    monthly_avg = (
        df[df['Country'] == country]
        .groupby('Month')['PM2.5']
        .mean()
        .round(2)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(x=monthly_avg.index, y=monthly_avg.values, marker='o', ax=ax)
    ax.set_title("📅 Monthly Average PM2.5 Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("PM2.5 µg/m³")
    ax.grid(True)
    st.pyplot(fig)

    # Download filtered data
    st.markdown("### 📄 Download Filtered Dataset")
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📅 Download CSV",
        data=csv,
        file_name="filtered_air_quality.csv",
        mime="text/csv"
    )

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
    
    @st.cache_data
    def train_model(model):
        model.fit(X, y)
        return model

    model = train_model(model)

    st.markdown("### 📅 Enter Feature Values")
    input_dict = {col: st.number_input(col, value=float(df[col].mean())) for col in feature_cols}
    input_array = np.array([list(input_dict.values())])
    predicted = model.predict(input_array)[0]

    # Health interpretation
    def interpret_pm25(value):
        if value <= 15:
            return "🟢 Good – Air quality is considered safe."
        elif value <= 35:
            return "🟡 Moderate – Acceptable, but some pollutants may pose a risk for sensitive groups."
        elif value <= 55:
            return "🟠 Unhealthy for Sensitive Groups – Limit prolonged outdoor exposure."
        elif value <= 150:
            return "🔴 Unhealthy – Everyone may begin to experience health effects."
        else:
            return "⚫ Very Unhealthy – Avoid outdoor activities."

    st.success(f"🌫️ Predicted PM2.5 Level using {model_choice}: {predicted:.2f} µg/m³")
    st.info(interpret_pm25(predicted))

Forecast Feature
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
