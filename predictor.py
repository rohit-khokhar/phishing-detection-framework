import joblib
import pandas as pd
import scipy.sparse as sp

from feature_engineering import (
    extract_url_features,
    extract_email_features,
    clean_for_tfidf
)

# ============================================
# Load Artifacts
# ============================================

EMAIL_MODEL = joblib.load(
    "deploy_artifacts/Email_StackingEnsemble.pkl"
)

URL_MODEL = joblib.load(
    "deploy_artifacts/URL_Lexical_StackingEnsemble.pkl"
)

TFIDF = joblib.load(
    "deploy_artifacts/tfidf_vectorizer_clean.pkl"
)

EMAIL_COLUMNS = joblib.load(
    "deploy_artifacts/email_feature_columns.pkl"
)

URL_COLUMNS = joblib.load(
    "deploy_artifacts/url_feature_columns.pkl"
)

# ============================================
# URL Prediction
# ============================================

def predict_url(url):

    features = extract_url_features(url)

    print("\nExtracted Features:")
    print(features)

    X = pd.DataFrame([features])
    X = X.reindex(columns=URL_COLUMNS, fill_value=0)

    prediction = URL_MODEL.predict(X)[0]
    probability = URL_MODEL.predict_proba(X)[0]

    print("Prediction:", prediction)
    print("Probability:", probability)

    return {
        "prediction": "Phishing" if prediction == 1 else "Legitimate",
        "confidence": round(max(probability) * 100, 2)
    }


# ============================================
# Email Prediction
# ============================================

def predict_email(email_text):

    handcrafted = extract_email_features(email_text)

    handcrafted_df = pd.DataFrame([handcrafted])

    handcrafted_df = handcrafted_df.reindex(
        columns=EMAIL_COLUMNS,
        fill_value=0
    )

    cleaned = clean_for_tfidf(email_text)

    tfidf_features = TFIDF.transform([cleaned])

    X = sp.hstack(
        [handcrafted_df.values, tfidf_features]
    )

    prediction = EMAIL_MODEL.predict(X)[0]

    probability = EMAIL_MODEL.predict_proba(X)[0]

    return {
        "prediction": "Phishing" if prediction == 1 else "Legitimate",
        "confidence": round(max(probability) * 100, 2)
    }


