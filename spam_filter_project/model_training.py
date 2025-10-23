"""
model_training.py
Train and save model; expose predict_proba helper with adjustable threshold.
"""
from sklearn.linear_model import LogisticRegression
import joblib

def train_basic_logreg(X_train, y_train):
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)
    return clf

def save_model(model, path='models/final_model.pkl'):
    joblib.dump(model, path)

def load_model(path='models/final_model.pkl'):
    return joblib.load(path)
