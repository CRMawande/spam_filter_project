# To run: streamlit run spam_filter_project/deployment.py
import streamlit as st
import joblib

@st.cache_resource
def load_assets():
    model = joblib.load('models/final_model.pkl')
    vec = joblib.load('models/vectorizer.pkl')
    return model, vec

model, vec = load_assets()

st.title("Spam Filter Demo")
msg = st.text_area("Enter SMS message here")
threshold = st.slider("Spam threshold (probability)", 0.0, 1.0, 0.5)

if st.button("Predict"):
    X = vec.transform([msg])
    prob = model.predict_proba(X)[0,1]
    st.write(f"Spam probability: {prob:.2f}")
    if prob >= threshold:
        st.warning("Message flagged as SPAM")
    else:
        st.success("Message appears HAM")
