
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.segment_model import load_data, analyze_segments

st.set_page_config(page_title="📊 Customer Segmentation Viewer", layout="centered")

st.title("📊 Customer Segmentation (Kaggle Dataset)")
st.write("Explore existing customer segments from your dataset.")

# Load data
data_path = "data/customers.csv"
try:
    data = load_data(data_path)
except FileNotFoundError:
    st.error("Please upload 'customers.csv' into the data folder.")
    st.stop()

st.subheader("🧾 Raw Data Preview")
st.dataframe(data.head())

# Segment analysis
st.subheader("📌 Segment Distribution")
segment_counts = analyze_segments(data)
st.bar_chart(segment_counts)

# State-wise segment breakdown
st.subheader("🌍 Segment Breakdown by State")
selected_segment = st.selectbox("Choose a segment:", segment_counts.index)
filtered = data[data["segment"] == selected_segment]

state_counts = filtered["state"].value_counts()
st.bar_chart(state_counts)

# City-wise display
st.subheader("🏙️ Top Cities in Selected Segment")
top_cities = filtered["city"].value_counts().head(10)
st.table(top_cities)
