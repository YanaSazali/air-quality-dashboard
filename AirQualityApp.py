# app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
page = st.selectbox("Select Page", ["Home", "Dashboard"])

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

    # Filters (moved to main content)
    st.markdown("### 🌐 Filter by Location")
    country = st.selectbox("Select Country", sorted(df['Country'].unique()))
    filtered_df = df[df['Country'] == country]
    cities = st.multiselect("Select Cities", sorted(filtered_df['City'].unique()), default=sorted(filtered_df['City'].unique())[:1])
    filtered_df = filtered_df[filtered_df['City'].isin(cities)]

    # Pollutant Selection
    pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    selected_pollutant = st.selectbox("Select Pollutant to Visualize", pollutants)

    # Line Chart
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

    # Distribution
    st.markdown(f"### 📊 Distribution of {selected_pollutant}")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(filtered_df, x=selected_pollutant, hue='City', kde=True, ax=ax, bins=30)
    st.pyplot(fig)

    # Data Table
    st.markdown("### 📄 Average Pollutant Levels by City")
    avg_pollutants = filtered_df.groupby('City')[pollutants].mean().round(2)
    st.dataframe(avg_pollutants)
