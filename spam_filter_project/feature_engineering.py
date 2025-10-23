"""
feature_engineering.py
TF-IDF and simple feature engineering helpers.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def fit_tfidf(corpus, max_features=5000):
    vec = TfidfVectorizer(stop_words='english', max_features=max_features, ngram_range=(1,2))
    X = vec.fit_transform(corpus)
    return vec, X

def save_vectorizer(vec, path='models/vectorizer.pkl'):
    joblib.dump(vec, path)
