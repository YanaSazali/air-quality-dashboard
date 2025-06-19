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

# Home Page
if page == "Home":
    st.markdown("""
        <h1 style='text-align: center; color: #FF4B4B;'>🌍 Welcome to the Air Quality Dashboard</h1>
        <h4 style='text-align: center; color: gray;'>Use the page selector above to explore air quality data by country and city</h4>
        <ul>
            <li>Filter data by country and city</li>
            <li>Visualize pollutant trends and distributions</li>
            <li>Download filtered datasets for analysis</li>
            <li>Use prediction tool to estimate PM2.5 levels</li>
        </ul>
        <p><strong>WHO Guideline:</strong> 24h PM2.5 should not exceed <strong>15 µg/m³</strong></p>
    """, unsafe_allow_html=True)

# Dashboard Page
elif page == "Dashboard":
    st.markdown("### 📊 Model Performance Comparison")

    @st.cache_data
    def train_models():
        X = df[[ 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed',
                 'Month', 'Day', 'Weekday', 'Is_Weekend', 'City_Mean_PM25', 'PM_Ratio',
                 'Humidity_Temp', 'O3_NO2', 'Lag_PM2.5', 'Rolling_PM2.5']]
        y = df['PM2.5']
        models = {
            'Linear Regression': Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())]),
            'Random Forest': Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor(n_estimators=100, random_state=42))]),
            'Decision Tree': Pipeline([('scaler', StandardScaler()), ('model', DecisionTreeRegressor(random_state=42))]),
            'Neural Network': Pipeline([('scaler', StandardScaler()), ('model', MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, early_stopping=True, random_state=42))])
        }
        results = {}
        for name, pipe in models.items():
            pipe.fit(X, y)
            y_pred = pipe.predict(X)
            results[name] = {
                'MAE': mean_absolute_error(y, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
                'R²': r2_score(y, y_pred)
            }
        return pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})

    model_results = train_models()
    st.dataframe(model_results)
    model_melt = model_results.melt(id_vars='Model', var_name='Metric', value_name='Score')
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=model_melt, x='Model', y='Score', hue='Metric', ax=ax)
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    st.pyplot(fig)

# Prediction Page
elif page == "Prediction":
    st.markdown("""
        <h1 style='text-align: center; color: #FF4B4B;'>🔮 Predict PM2.5 Levels</h1>
        <h4 style='text-align: center; color: gray;'>Estimate PM2.5 concentration using multiple features</h4>
    """, unsafe_allow_html=True)

    feature_cols = [ 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind Speed',
                     'Month', 'Day', 'Weekday', 'Is_Weekend', 'City_Mean_PM25', 'PM_Ratio',
                     'Humidity_Temp', 'O3_NO2', 'Lag_PM2.5', 'Rolling_PM2.5']
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

        st.markdown("### 📊 Model Comparison on Current Data")

    @st.cache_data
    def compare_models(X, y):
        models = {
            'Linear Regression': Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())]),
            'Random Forest': Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor(n_estimators=100, random_state=42))]),
            'Decision Tree': Pipeline([('scaler', StandardScaler()), ('model', DecisionTreeRegressor(random_state=42))]),
            'Neural Network': Pipeline([('scaler', StandardScaler()), ('model', MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, early_stopping=True, random_state=42))])
        }

        results = {}
        for name, pipe in models.items():
            pipe.fit(X, y)
            y_pred = pipe.predict(X)
            results[name] = {
                'MAE': mean_absolute_error(y, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
                'R²': r2_score(y, y_pred)
            }

        return pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})

    model_metrics = compare_models(X, y)
    st.dataframe(model_metrics)

    # Optional: Add bar plot
    plot_df = model_metrics.melt(id_vars="Model", var_name="Metric", value_name="Value")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_df, x="Model", y="Value", hue="Metric", ax=ax)
    ax.set_title("Model Evaluation Metrics")
    ax.set_ylabel("Score")
    st.pyplot(fig)


    st.markdown("### 📥 Enter Feature Values")
    input_dict = {col: st.number_input(col, value=float(df[col].mean())) for col in feature_cols}
    input_array = np.array([list(input_dict.values())])
    predicted = model.predict(input_array)[0]

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

   
