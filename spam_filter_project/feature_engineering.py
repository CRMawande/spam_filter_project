import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import joblib

def load_preprocessed_data(data_path):
    df = pd.read_csv(data_path)
    return df

def vectorize_text_tfidf(df, max_features=5000):
    # Fill empty/NaN processed messages
    df['processed_message'] = df['processed_message'].fillna('empty').replace('', 'empty')
    # TF-IDF for LR/SVC/CNB
    vectorizer_tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_text_tfidf = vectorizer_tfidf.fit_transform(df['processed_message'])
    return X_text_tfidf, vectorizer_tfidf

def vectorize_text_count(df, max_features=5000):
    # Count Vectorizer for MNB
    df['processed_message'] = df['processed_message'].fillna('empty').replace('', 'empty')
    vectorizer_count = CountVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_text_count = vectorizer_count.fit_transform(df['processed_message'])
    return X_text_count, vectorizer_count

# --- CHANGED: scale_numeric_feature now uses MinMaxScaler ---
def scale_numeric_feature(df):
    # MinMax scaling forces values to be non-negative [0, 1]
    scaler = MinMaxScaler()
    X_length_scaled = scaler.fit_transform(df[['message_length']])
    return X_length_scaled, scaler

def combine_features(X_text, X_length_scaled):
    return hstack((X_text, X_length_scaled))

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def engineer_features(data_path, output_dir):
    df = load_preprocessed_data(data_path)
    y = df['label_num']
    
    # 1. TF-IDF Features 
    X_text_tfidf, vectorizer_tfidf = vectorize_text_tfidf(df)
    
    # 2. Count Features 
    X_text_count, vectorizer_count = vectorize_text_count(df)
    
    # 3. Numeric Feature Scaling (MinMax for compatibility)
    X_length_scaled, scaler = scale_numeric_feature(df)
    
    # Combine both feature sets with the non-negative length feature
    X_tfidf_combined = combine_features(X_text_tfidf, X_length_scaled)
    X_count_combined = combine_features(X_text_count, X_length_scaled)
    
    # Split TF-IDF (used for LR, SVC, CNB)
    X_train_tfidf, X_test_tfidf, y_train, y_test = split_data(X_tfidf_combined, y)
    
    # Split Count (used for MNB)
    X_train_count, X_test_count, _, _ = split_data(X_count_combined, y) 
    
    # 4. Save All Assets ---
    os.makedirs(output_dir, exist_ok=True)
    
    # TF-IDF Assets
    joblib.dump(vectorizer_tfidf, os.path.join(output_dir, 'vectorizer_tfidf.pkl'))
    joblib.dump(X_train_tfidf, os.path.join(output_dir, 'X_train_tfidf.pkl'))
    joblib.dump(X_test_tfidf, os.path.join(output_dir, 'X_test_tfidf.pkl'))

    # Count Assets
    joblib.dump(vectorizer_count, os.path.join(output_dir, 'vectorizer_count.pkl'))
    joblib.dump(X_train_count, os.path.join(output_dir, 'X_train_count.pkl'))
    joblib.dump(X_test_count, os.path.join(output_dir, 'X_test_count.pkl'))
    
    # Common Assets
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(y_train, os.path.join(output_dir, 'y_train.pkl'))
    joblib.dump(y_test, os.path.join(output_dir, 'y_test.pkl'))
    
    print(f"[FEATURES] TF-IDF Features: {X_tfidf_combined.shape[1]}")
    print(f"[FEATURES] Count Features: {X_count_combined.shape[1]}")
    print(f"Training set size: {X_train_tfidf.shape[0]}, Test set size: {X_test_tfidf.shape[0]}")

if __name__ == '__main__':
    print("--- Testing feature_engineering.py ---")
    DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'preprocessed_data.csv'))
    OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    
    try:
        engineer_features(DATA_PATH, OUTPUT_DIR)
    except FileNotFoundError:
        print(f"TEST FAILED: Please ensure 'preprocessed_data.csv' is in the 'data/processed/' directory at: {DATA_PATH}")
        exit()