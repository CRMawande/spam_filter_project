"""
model_evaluation.py
Evaluation utilities.
"""
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))
