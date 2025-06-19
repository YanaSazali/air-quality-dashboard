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
import plotly.express as px  # For interactive maps

# Set page config (mobile-friendly)
st.set_page_config(
    page_title="Air Quality Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for mobile responsiveness and better visuals
st.markdown("""
    <style>
    body {
        background-color: #f8f9fa;
        font-family: Arial, sans-serif;
    }
    .stSelectbox, .stMultiselect, .stSlider, .stNumberInput {
        margin-bottom: 1rem;
    }
    @media (max-width: 768px) {
        .stPlotlyChart, .stDataFrame {
            width: 100% !important;
        }
    }
    .health-alert {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .good { background-color: #4CAF50; color: white; }
    .moderate { background-color: #FFEB3B; color: black; }
    .unhealthy-sensitive { background-color: #FF9800; color: white; }
    .unhealthy { background-color: #F44336; color: white; }
    .very-unhealthy { background-color: #9C27B0; color: white; }
    </style>
""", unsafe_allow_html=True)

# Load and clean data
@st.cache_resource
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

# AQI Calculator (US EPA standard for PM2.5)
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

# Health interpretation with CSS classes
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

# Policy simulation: Adjust pollutants and predict PM2.5 impact
def simulate_policy_change(base_values, adjustments):
    adjusted_values = base_values.copy()
    for pollutant, change in adjustments.items():
        adjusted_values[pollutant] *= (1 + change/100)
    return adjusted_values

# Page selector
page = st.sidebar.selectbox("Select Page", ["Home", "Dashboard", "Prediction", "Policy Simulation"])

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

elif page == "Dashboard":
    st.markdown("""
        <h1 style='text-align: center; color: #FF4B4B;'>Air Quality Dashboard</h1>
        <h4 style='text-align: center; color: gray;'>Track & Visualize Global Pollutant Levels</h4>
    """, unsafe_allow_html=True)

    # Filters
    st.sidebar.markdown("### 🔍 Filters")
    country = st.sidebar.selectbox("Select Country", sorted(df['Country'].unique()))
    filtered_df = df[df['Country'] == country]
    cities = st.sidebar.multiselect(
        "Select Cities", 
        sorted(filtered_df['City'].unique()), 
        default=sorted(filtered_df['City'].unique())[:1]
    )
    pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    selected_pollutant = st.sidebar.selectbox("Select Pollutant", pollutants)

    # Filter data
    filtered_df = filtered_df[filtered_df['City'].isin(cities)]

    # Map Visualization
    st.markdown("### City Locations")
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
    fig = px.scatter_mapbox(
        map_df, lat="Lat", lon="Lon", hover_name="City", size="PM2.5",
        color="PM2.5", zoom=5, height=300,
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

    # Time Series Plot
    st.markdown(f"### {selected_pollutant} Over Time")
    fig, ax = plt.subplots(figsize=(12, 5))
    for city in cities:
        city_data = filtered_df[filtered_df['City'] == city]
        city_avg = city_data.groupby('Date')[selected_pollutant].mean()
        ax.plot(city_avg.index, city_avg.values, label=city)
    ax.set_ylabel(f"{selected_pollutant} (µg/m³)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Health Alerts
    st.markdown("### ⚠️ Current Health Alerts")
    for city in cities:
        avg_pm25 = filtered_df[filtered_df['City'] == city]['PM2.5'].mean()
        status, message, css_class = interpret_pm25(avg_pm25)
        st.markdown(f"""
            <div class="health-alert {css_class}">
                <b>{city}: {status}</b> – {message} (Avg PM2.5: {avg_pm25:.1f} µg/m³)
            </div>
        """, unsafe_allow_html=True)

    # Download Data
    st.sidebar.markdown("### Download Data")
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"air_quality_{country}.csv",
        mime="text/csv"
    )

elif page == "Prediction":
    st.markdown("""
        <h1 style='text-align: center; color: #FF4B4B;'>Predict PM2.5 Levels</h1>
        <h4 style='text-align: center; color: gray;'>Estimate PM2.5 concentration using machine learning</h4>
    """, unsafe_allow_html=True)

    # Model selection
    model_choice = st.selectbox(
        "Choose Model", 
        ["Linear Regression", "Random Forest", "Decision Tree", "Neural Network"],
        help="Random Forest is recommended for accuracy"
    )

    # Input sliders
    st.markdown("### Enter Environmental Conditions")
    col1, col2 = st.columns(2)
    with col1:
        pm10 = st.slider("PM10 (µg/m³)", 0.0, 200.0, float(df['PM10'].mean()))
        no2 = st.slider("NO2 (µg/m³)", 0.0, 100.0, float(df['NO2'].mean()))
        temp = st.slider("Temperature (°C)", -10.0, 40.0, float(df['Temperature'].mean()))
        humidity = st.slider("Humidity (%)", 0.0, 100.0, float(df['Humidity'].mean()))
    with col2:
        co = st.slider("CO (µg/m³)", 0.0, 10.0, float(df['CO'].mean()))
        o3 = st.slider("O3 (µg/m³)", 0.0, 200.0, float(df['O3'].mean()))
        wind = st.slider("Wind Speed (m/s)", 0.0, 20.0, float(df['Wind Speed'].mean()))
        city_mean = st.slider("City Avg PM2.5", 0.0, 150.0, float(df['City_Mean_PM25'].mean()))

    # Prepare input
    input_data = pd.DataFrame({
        'PM10': [pm10],
        'NO2': [no2],
        'SO2': [df['SO2'].mean()],  # Default values for unused features
        'CO': [co],
        'O3': [o3],
        'Temperature': [temp],
        'Humidity': [humidity],
        'Wind Speed': [wind],
        'City_Mean_PM25': [city_mean],
        'PM_Ratio': [pm10 / (pm10 * 0.5 + 1e-5)],  # Approximation
        'Humidity_Temp': [humidity * temp],
        'O3_NO2': [o3 / (no2 + 1e-5)],
        'Month': [6],  # Default month (June)
        'Day': [15],
        'Weekday': [2],
        'Is_Weekend': [0],
        'Lag_PM2.5': [city_mean * 0.9],  # Approximation
        'Rolling_PM2.5': [city_mean]
    })

    # Train model
    feature_cols = input_data.columns.tolist()
    X = df[feature_cols]
    y = df['PM2.5']

    if model_choice == "Linear Regression":
        model = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
    elif model_choice == "Random Forest":
        model = Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor(n_estimators=100, random_state=42))])
    elif model_choice == "Decision Tree":
        model = Pipeline([('scaler', StandardScaler()), ('model', DecisionTreeRegressor(random_state=42))])
    elif model_choice == "Neural Network":
        model = Pipeline([('scaler', StandardScaler()), ('model', MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=42))])

    model.fit(X, y)
    prediction = model.predict(input_data)[0]

    # Display results
    st.markdown("### Prediction Results")
    aqi = calculate_aqi(prediction)
    status, message, css_class = interpret_pm25(prediction)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted PM2.5", f"{prediction:.1f} µg/m³")
    with col2:
        st.metric("Air Quality Index (AQI)", f"{aqi:.0f}")

    st.markdown(f"""
        <div class="health-alert {css_class}">
            <b>Health Impact: {status}</b> – {message}
        </div>
    """, unsafe_allow_html=True)

    # Actionable recommendations
    st.markdown("### 📋 Recommended Actions")
    if prediction <= 35.4:
        st.success("✅ No significant health risks. Maintain current activities.")
    elif prediction <= 55.4:
        st.warning("⚠️ Sensitive groups should reduce prolonged outdoor exertion.")
    else:
        st.error("🚫 Everyone should avoid outdoor activities. Close windows and use air purifiers.")

elif page == "Policy Simulation":
    st.markdown("""
        <h1 style='text-align: center; color: #FF4B4B;'>🏛️ Policy Simulation</h1>
        <h4 style='text-align: center; color: gray;'>Test how emission reductions affect PM2.5 levels</h4>
    """, unsafe_allow_html=True)

    # Baseline inputs
    st.markdown("### 📜 Baseline Pollution Levels")
    col1, col2 = st.columns(2)
    with col1:
        pm10_base = st.number_input("PM10 (µg/m³)", value=float(df['PM10'].mean()))
        no2_base = st.number_input("NO2 (µg/m³)", value=float(df['NO2'].mean()))
    with col2:
        co_base = st.number_input("CO (µg/m³)", value=float(df['CO'].mean()))
        o3_base = st.number_input("O3 (µg/m³)", value=float(df['O3'].mean()))

    # Policy adjustments
    st.markdown("### ⚙️ Policy Adjustments (% Change)")
    pm10_change = st.slider("PM10 Reduction (%)", -50, 50, 0, help="Negative values = increase")
    no2_change = st.slider("NO2 Reduction (%)", -50, 50, 0)
    co_change = st.slider("CO Reduction (%)", -50, 50, 0)
    o3_change = st.slider("O3 Reduction (%)", -50, 50, 0)

    # Simulate
    if st.button("Simulate Policy Impact"):
        adjustments = {
            'PM10': -pm10_change,  # Negative because reduction means lower PM10
            'NO2': -no2_change,
            'CO': -co_change,
            'O3': -o3_change
        }
        simulated_values = simulate_policy_change(
            {'PM10': pm10_base, 'NO2': no2_base, 'CO': co_base, 'O3': o3_base},
            adjustments
        )
        
        # Predict PM2.5 (using simplified model)
        model = RandomForestRegressor()
        X = df[['PM10', 'NO2', 'CO', 'O3']]
        y = df['PM2.5']
        model.fit(X, y)
        
        baseline_pm25 = model.predict([[pm10_base, no2_base, co_base, o3_base]])[0]
        new_pm25 = model.predict([
            [simulated_values['PM10'], simulated_values['NO2'], 
             simulated_values['CO'], simulated_values['O3']]
        ])[0]
        
        # Display results
        st.markdown("### 📊 Simulation Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Baseline PM2.5", f"{baseline_pm25:.1f} µg/m³")
        with col2:
            st.metric("New PM2.5", f"{new_pm25:.1f} µg/m³", delta=f"{(new_pm25 - baseline_pm25):.1f} µg/m³")
        
        # Interpretation
        st.markdown("#### 💡 Policy Impact Summary")
        if new_pm25 < baseline_pm25:
            st.success(f"✅ This policy could reduce PM2.5 by {(baseline_pm25 - new_pm25):.1f} µg/m³")
        else:
            st.warning(f"⚠️ This policy may increase PM2.5 by {(new_pm25 - baseline_pm25):.1f} µg/m³")
