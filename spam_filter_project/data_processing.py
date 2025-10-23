"""
data_processing.py
Simple, reusable ETL helpers (place final versions here).
"""
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', header=None, names=['label','text'])
    return df

def split_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    X = df['text']
    y = df['label'].map({'ham':0, 'spam':1})
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
