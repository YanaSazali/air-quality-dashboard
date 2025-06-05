# app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Air Quality Dashboard", layout="wide")

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

st.title("Interactive Air Quality Dashboard")

# Sidebar Filters
st.sidebar.header("Filter by Location")
country = st.sidebar.selectbox("Select Country", sorted(df['Country'].unique()))

filtered_df = df[df['Country'] == country]

cities = st.sidebar.multiselect(
    "Select Cities",
    sorted(filtered_df['City'].unique()),
    default=sorted(filtered_df['City'].unique())[:1]
)

filtered_df = filtered_df[filtered_df['City'].isin(cities)]

# Main Display
st.subheader(f"Air Quality for {', '.join(cities)} in {country}")

# Time series and pollutant comparison
pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
selected_pollutant = st.selectbox("Select Pollutant to Visualize", pollutants)

# Time series plot
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

# Distribution plot
st.markdown(f"### Distribution of {selected_pollutant}")
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(filtered_df, x=selected_pollutant, hue='City', kde=True, ax=ax, bins=30)
st.pyplot(fig)

# Average pollutant levels per city
st.markdown("### Average Pollutant Levels by City")
avg_pollutants = filtered_df.groupby('City')[pollutants].mean().round(2)
st.dataframe(avg_pollutants)

# Optional: Export filtered data
st.download_button(
    label="Download Filtered Data as CSV",
    data=filtered_df.to_csv(index=False),
    file_name=f"air_quality_{country}_{'_'.join(cities)}.csv",
    mime='text/csv'
)
