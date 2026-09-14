"""
Credit Card Fraud Detection - Streamlit Interface
--------------------------------------------------
This app ONLY loads the already-trained model (fraud_detection_model.pkl)
and uses it for prediction. No ML/model logic, preprocessing, or training
code has been modified, retrained, or rewritten.

Run with:
    streamlit run app.py
"""

# =========================================================
# 1. IMPORTS
# =========================================================
import os
import joblib
import pandas as pd
import streamlit as st


# =========================================================
# 2. MODEL LOADING
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fraud_detection_model.pkl")


@st.cache_resource
def load_model(path: str):
    """Load the pre-trained fraud detection pipeline (ColumnTransformer + XGBoost)."""
    if not os.path.exists(path):
        return None, f"Model file not found at: {path}"
    try:
        model = joblib.load(path)
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {e}"


model, load_error = load_model(MODEL_PATH)


# =========================================================
# 3. STREAMLIT PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---- Light theme styling (inline CSS via st.markdown) ----
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .stApp {
            background-color: #f7f8fc;
        }

        /* Hide default streamlit chrome for a cleaner look */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Header banner */
        .app-header {
            background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
            padding: 2rem 2.25rem;
            border-radius: 18px;
            margin-bottom: 1.75rem;
            box-shadow: 0 10px 25px -8px rgba(79, 70, 229, 0.45);
        }
        .app-header h1 {
            color: #ffffff;
            font-size: 1.9rem;
            font-weight: 800;
            margin: 0;
        }
        .app-header p {
            color: #e0e7ff;
            font-size: 0.98rem;
            margin-top: 0.4rem;
            margin-bottom: 0;
        }

        /* Generic card */
        .card {
            background: #ffffff;
            border: 1px solid #edeff5;
            border-radius: 16px;
            padding: 1.6rem 1.75rem;
            box-shadow: 0 2px 10px rgba(20, 20, 50, 0.04);
            margin-bottom: 1.25rem;
        }
        .card h3 {
            margin-top: 0;
            font-weight: 700;
            color: #1e1b4b;
            font-size: 1.15rem;
        }

        /* Section labels */
        .section-label {
            font-weight: 600;
            color: #4338ca;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }

        /* Result cards */
        .result-fraud {
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
            border: 1.5px solid #fca5a5;
            border-radius: 16px;
            padding: 1.75rem;
            text-align: center;
        }
        .result-fraud h2 {
            color: #b91c1c;
            font-size: 1.6rem;
            font-weight: 800;
            margin: 0.3rem 0;
        }
        .result-legit {
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border: 1.5px solid #86efac;
            border-radius: 16px;
            padding: 1.75rem;
            text-align: center;
        }
        .result-legit h2 {
            color: #15803d;
            font-size: 1.6rem;
            font-weight: 800;
            margin: 0.3rem 0;
        }
        .result-icon {
            font-size: 2.6rem;
        }
        .result-sub {
            color: #52525b;
            font-size: 0.9rem;
            margin-top: 0.25rem;
        }

        /* Buttons */
        div.stButton > button {
            background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
            color: white;
            font-weight: 700;
            border: none;
            border-radius: 10px;
            padding: 0.7rem 1rem;
            width: 100%;
            font-size: 1rem;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
            transition: all 0.15s ease-in-out;
        }
        div.stButton > button:hover {
            background: linear-gradient(135deg, #4338ca 0%, #4f46e5 100%);
            box-shadow: 0 6px 16px rgba(79, 70, 229, 0.4);
        }

        /* Footer note */
        .footer-note {
            text-align: center;
            color: #9ca3af;
            font-size: 0.8rem;
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 4. UI - HEADER
# =========================================================
st.markdown(
    """
    <div class="app-header">
        <h1>💳 Credit Card Fraud Detection</h1>
        <p>Enter transaction details to check whether it looks fraudulent or legitimate, using a trained XGBoost pipeline.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Stop early with a clear error if the model could not be loaded
if load_error:
    st.error(f"⚠️ {load_error}\n\nMake sure 'fraud_detection_model.pkl' is placed in the same folder as app.py.")
    st.stop()


# =========================================================
# 5. USER INPUT COLLECTION
# =========================================================
left_col, right_col = st.columns([1.15, 0.85], gap="large")

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧾 Transaction Details")

    c1, c2 = st.columns(2)
    with c1:
        amount = st.number_input(
            "Transaction Amount ($)",
            min_value=0.0,
            max_value=100000.0,
            value=100.0,
            step=1.0,
            help="Total transaction amount in dollars.",
        )
        transaction_hour = st.slider(
            "Transaction Hour (0-23)",
            min_value=0,
            max_value=23,
            value=12,
            help="Hour of the day the transaction occurred (24-hour format).",
        )
        merchant_category = st.selectbox(
            "Merchant Category",
            options=["Electronics", "Travel", "Grocery", "Food", "Clothing"],
        )
        foreign_transaction_label = st.selectbox(
            "Foreign Transaction",
            options=["No", "Yes"],
            help="Was this transaction made in a foreign country?",
        )

    with c2:
        location_mismatch_label = st.selectbox(
            "Location Mismatch",
            options=["No", "Yes"],
            help="Does the transaction location differ from the cardholder's usual location?",
        )
        device_trust_score = st.slider(
            "Device Trust Score (0-100)",
            min_value=0,
            max_value=100,
            value=65,
            help="Trust score of the device used for the transaction.",
        )
        velocity_last_24h = st.number_input(
            "Velocity in Last 24 Hours",
            min_value=0,
            max_value=100,
            value=2,
            step=1,
            help="Number of transactions made by this card in the last 24 hours.",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    predict_clicked = st.button("🔍 Predict Fraud", use_container_width=True)

# Convert Yes/No labels back to the 0/1 encoding used during training
foreign_transaction = 1 if foreign_transaction_label == "Yes" else 0
location_mismatch = 1 if location_mismatch_label == "Yes" else 0


# =========================================================
# 6. PREDICTION
# =========================================================
with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Prediction Result")

    if predict_clicked:
        try:
            # Build a single-row DataFrame with EXACTLY the feature names
            # the trained pipeline (ColumnTransformer) expects.
            input_df = pd.DataFrame(
                [{
                    "amount": amount,
                    "transaction_hour": transaction_hour,
                    "merchant_category": merchant_category,
                    "foreign_transaction": foreign_transaction,
                    "location_mismatch": location_mismatch,
                    "device_trust_score": device_trust_score,
                    "velocity_last_24h": velocity_last_24h,
                }]
            )

            prediction = model.predict(input_df)[0]

            fraud_probability = None
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(input_df)[0]
                    # Assume class "1" (fraud) is the positive class
                    fraud_probability = float(proba[1])
                except Exception:
                    fraud_probability = None

            if int(prediction) == 1:
                st.markdown(
                    """
                    <div class="result-fraud">
                        <div class="result-icon">🚨</div>
                        <h2>Fraudulent Transaction</h2>
                        <div class="result-sub">This transaction has been flagged as high-risk.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="result-legit">
                        <div class="result-icon">✅</div>
                        <h2>Legitimate Transaction</h2>
                        <div class="result-sub">This transaction appears to be safe.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if fraud_probability is not None:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">Fraud Probability</div>', unsafe_allow_html=True)
                st.progress(min(max(fraud_probability, 0.0), 1.0))
                st.write(f"**{fraud_probability * 100:.2f}%** estimated probability of fraud")

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
    else:
        st.info("Fill in the transaction details and click **Predict Fraud** to see the result here.")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="footer-note">Built with Streamlit · Model: XGBoost Classifier (ColumnTransformer + StandardScaler + OneHotEncoder)</div>',
    unsafe_allow_html=True,
)
