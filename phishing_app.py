# phishing_app.py

import streamlit as st
import joblib
import pandas as pd
from features.feature_extractor import extract_url_features

# List of features your model was trained on
selected_features = [
    "URLSimilarityIndex",
    "LetterRatioInURL",
    "IsHTTPS",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "TLDLegitimateProb",
    "url_entropy",
    "CharContinuationRate",
    "DegitRatioInURL",
    "NoOfDegitsInURL",
    "URLLength",
    "NoOfLettersInURL",
    "DomainEntropy",
    "URLCharProb",
    "path_depth",
    "DomainLength",
    "NoOfSubDomain",
    "Pay",
    "TLDLength",
    "TLD_reduced_app",
    "NoOfQMarkInURL",
    "Bank",
    "TLD_reduced_com",
    "NoOfEqualsInURL",
    "TLD_reduced_org",
    "has_login_keyword",
    "TLD_reduced_co",
    "has_bank_or_pay_keyword",
    "TLD_reduced_io",
    "TLD_reduced_uk"
]

# Load the model (make sure the path is correct)
@st.cache_resource
def load_model():
    return joblib.load("phishing_url_detector_xgb3.pkl")

model = load_model()

# UI
st.title("🔗 Phishing URL Detector")

st.markdown("""
This app uses a trained XGBoost model to predict whether a URL is **phishing** or **legitimate**.
""")

url_input = st.text_input("Paste a URL below:", placeholder="https://example.com/login")

if st.button("Check URL"):
    if not url_input.strip():
        st.warning("Please enter a URL.")
    else:
        try:
            # Extract numeric features in the same order as training
            numeric_feats = extract_url_features(url_input)
            if len(numeric_feats) != len(selected_features):
                raise ValueError("Feature length mismatch")

            # Build DataFrame with exactly the same feature names
            X_new = pd.DataFrame([numeric_feats], columns=selected_features)

            # Predict
            prob = model.predict_proba(X_new)[:, 1][0]
            pred = model.predict(X_new)[0]

            # Display
            st.divider()
            st.write("### Prediction")

            if pred == 1:
                st.error(f"🔴 **Phishing URL (high risk)**", icon="🚨")
            else:
                st.success(f"✅ **Likely legitimate**", icon="✅")

            st.write(f"**Phishing probability:** {prob:.3f}")
            st.code(url_input)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
