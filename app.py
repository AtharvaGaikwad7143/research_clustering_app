import streamlit as st
import pandas as pd
from cluster_model import load_data, preprocess_text, cluster_texts, get_top_keywords_per_cluster

# App configuration
st.set_page_config(page_title="📚 Research Article Clustering", layout="centered")
st.title("📚 Research Paper Clustering using KMeans")

# File upload section
uploaded_file = st.file_uploader("Upload a CSV with 'title' and 'abstract' columns", type=["csv"])
if not uploaded_file:
    st.info("Please upload a CSV file to proceed")
    st.stop()

# Load and preview data
try:
    data = load_data(uploaded_file)
    if 'abstract' not in data.columns:
        st.error("Missing required column: 'abstract'")
        st.stop()

    st.subheader("🔍 Raw Data Preview")
    st.dataframe(data.head())

    # Preprocessing
    texts = preprocess_text(data)

    # Clustering configuration
    st.subheader("🔢 Choose Number of Clusters")
    k = st.slider("Clusters (k)", 2, 10, 5)
    
    # Perform clustering
    clusters, model, vectorizer = cluster_texts(texts, k)
    data["Cluster"] = clusters

    # Visualizations
    st.subheader("📊 Cluster Distribution")
    st.bar_chart(data["Cluster"].value_counts())

    # Top keywords
    st.subheader("💡 Top Keywords Per Cluster")
    keywords = get_top_keywords_per_cluster(model, vectorizer)
    for cluster, words in keywords.items():
        st.markdown(f"**{cluster}:** " + ", ".join(words))

    # Filter papers by cluster
    st.subheader("📂 View Articles in Cluster")
    selected = st.selectbox("Choose a Cluster", sorted(data["Cluster"].unique()))
    st.write(data[data["Cluster"] == selected][["title", "abstract"]])

except Exception as e:
    st.error(f"An error occurred: {str(e)}")