import pandas as pd
import nltk
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Safe stopwords download if not available
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
stopwords_path = os.path.join(nltk_data_path, "corpora", "stopwords")
if not os.path.exists(stopwords_path):
    nltk.download("stopwords", download_dir=nltk_data_path)

from nltk.corpus import stopwords

def load_data(path):
    """Load CSV data into pandas DataFrame"""
    return pd.read_csv(path)

def preprocess_text(data, column='abstract'):
    """Preprocess text by removing stopwords and non-alphabetic characters"""
    stop_words = set(stopwords.words('english'))
    return data[column].fillna("").apply(lambda x: ' '.join(
        [word for word in x.lower().split() if word.isalpha() and word not in stop_words]
    ))

def cluster_texts(texts, num_clusters=5):
    """Cluster texts using KMeans and TF-IDF vectorization"""
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts)
    model = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = model.fit_predict(X)
    return clusters, model, vectorizer

def get_top_keywords_per_cluster(model, vectorizer, num_keywords=10):
    """Extract top keywords for each cluster"""
    keywords = {}
    terms = vectorizer.get_feature_names_out()
    for i, center in enumerate(model.cluster_centers_):
        indices = center.argsort()[-num_keywords:][::-1]
        keywords[f"Cluster {i}"] = [terms[idx] for idx in indices]
    return keywords
