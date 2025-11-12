import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack

# CONFIG
MODELS_DIR = "models"
REPORTS_DIR = "reports"

# LOAD ARTIFACTS
best_mnb = joblib.load(os.path.join(MODELS_DIR, "MNB_best.pkl"))
vectorizer_count = joblib.load(os.path.join(MODELS_DIR, "vectorizer_count.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
X_test_count = joblib.load(os.path.join(MODELS_DIR, "X_test_count.pkl"))
y_test = joblib.load(os.path.join(MODELS_DIR, "y_test.pkl"))
high_precision_threshold = joblib.load(os.path.join(MODELS_DIR, "high_precision_threshold.pkl"))

# STREAMLIT UI
st.set_page_config(page_title="Adjustable Risk Spam Filter", layout="centered")
st.title("📩 Adjustable Risk Spam Filter")
st.markdown("Drag the slider to adjust the strictness of the spam filter:")

# RISK LEVELS
RISK_MAPPING = {
    "High-Risk (lenient)": 0.3,
    "Medium-Risk (balanced)": 0.5,
    "Low-Risk (strict)": float(high_precision_threshold)
}

# Slider for threshold
threshold = st.slider(
    "Adjust Threshold",
    min_value=0.0, max_value=1.0, value=0.5, step=0.01
)

# Display risk level dynamically
if threshold >= RISK_MAPPING["Low-Risk (strict)"]:
    risk_label = "Low-Risk (strict)"
elif threshold >= RISK_MAPPING["Medium-Risk (balanced)"]:
    risk_label = "Medium-Risk (balanced)"
else:
    risk_label = "High-Risk (lenient)"

st.markdown(f"**Selected Risk Level:** {risk_label} (Threshold: {threshold:.3f})")

# USER INPUT ===
user_input = st.text_area("Enter a message to test:", placeholder="Type or paste a message here...")

if st.button("Classify Message"):
    if not user_input.strip():
        st.warning("Please enter a message before classifying.")
    else:
        X_input = vectorizer_count.transform([user_input])
        X_length = np.array([[len(user_input)]])
        X_length_scaled = scaler.transform(X_length)
        X_combined = hstack((X_input, X_length_scaled))

        prob_spam = best_mnb.predict_proba(X_combined)[:, 1][0]
        label = int(prob_spam >= threshold)

        st.markdown("###Prediction Result")
        if label == 1:
            st.error(f"Classified as **SPAM** (Probability: {prob_spam:.4f})")
        else:
            st.success(f"Classified as **HAM** (Probability: {prob_spam:.4f})")

# === SIMULATED REVIEW FOLDER IMPACT (COMPACT) ===
st.markdown("---")
st.subheader("Simulation: Review Folder Impact (Compact)")

y_scores = best_mnb.predict_proba(X_test_count)[:, 1]
simulation = []
for name, th in RISK_MAPPING.items():
    y_pred = (y_scores >= th).astype(int)
    review_count = np.sum(y_pred == 1)
    passed_count = np.sum(y_pred == 0)
    total = len(y_test)
    simulation.append({
        "Risk Level": name,
        "Threshold": round(th, 3),
        "Spam (Review Folder)": review_count,
        "Passed Messages": passed_count,
        "Review %": round(review_count / total * 100, 1)
    })

sim_df = pd.DataFrame(simulation)
st.dataframe(sim_df, use_container_width=True, height=200)
