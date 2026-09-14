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
import math
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
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---- small inline icon set (stroke-based, currentColor) ----
ICONS = {
    "amount": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="2" x2="12" y2="22"/><path d="M17 6.5c0-1.9-2.2-3.5-5-3.5S7 4.6 7 6.5 9.2 10 12 10s5 1.6 5 3.5-2.2 3.5-5 3.5-5-1.6-5-3.5"/></svg>',
    "clock": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3.2 2"/></svg>',
    "store": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 9V5h16v4"/><path d="M3 9h18l-1.2 3.5a2 2 0 0 1-1.9 1.5H6.1a2 2 0 0 1-1.9-1.5L3 9z"/><path d="M5 14v6h14v-6"/><path d="M10 20v-4h4v4"/></svg>',
    "globe": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.5 2.6 3.8 5.8 3.8 9s-1.3 6.4-3.8 9c-2.5-2.6-3.8-5.8-3.8-9S9.5 5.6 12 3z"/></svg>',
    "pin": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s7-7.4 7-12.5A7 7 0 0 0 5 9.5C5 14.6 12 22 12 22z"/><circle cx="12" cy="9.5" r="2.4"/></svg>',
    "signal": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 20V12M10 20V8M16 20V4M22 20H2"/></svg>',
    "repeat": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M17 2l4 4-4 4"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><path d="M7 22l-4-4 4-4"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>',
    "shield": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2l8 3v6c0 5-3.4 8.7-8 11-4.6-2.3-8-6-8-11V5l8-3z"/><path d="M9 12l2 2 4-4"/></svg>',
    "alert": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M10.3 3.9L2.6 17a2 2 0 0 0 1.7 3h15.4a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><line x1="12" y1="9" x2="12" y2="13.5"/><line x1="12" y1="16.7" x2="12" y2="16.9"/></svg>',
    "radar": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.2" fill="currentColor" stroke="none"/><path d="M12 3v3M21 12h-3"/></svg>',
    "dot": '<svg viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="6"/></svg>',
}


def icon(name: str, size: int = 18) -> str:
    return f'<span style="display:inline-flex;width:{size}px;height:{size}px;vertical-align:-4px;">{ICONS[name]}</span>'


# ---- theme styling ----
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp { background: #F3F5F9; }
        #MainMenu, footer, header { visibility: hidden; }
        .block-container { padding-top: 1.6rem; max-width: 1180px; }

        h1, h2, h3, .mono { font-family: 'Space Grotesk', sans-serif; }
        .mono-data { font-family: 'JetBrains Mono', monospace; }

        /* ---------- header ---------- */
        .app-header {
            position: relative;
            background: radial-gradient(circle at 18% 20%, #1E2A52 0%, #0B1220 62%);
            border-radius: 20px;
            padding: 2.1rem 2.4rem;
            margin-bottom: 1.6rem;
            overflow: hidden;
            box-shadow: 0 18px 40px -18px rgba(11, 18, 32, 0.55);
        }
        .app-header::before {
            content: "";
            position: absolute; inset: 0;
            background-image: radial-gradient(rgba(148,163,255,0.16) 1px, transparent 1px);
            background-size: 16px 16px;
            mask-image: linear-gradient(to right, black, transparent 75%);
        }
        .header-row { position: relative; display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; flex-wrap: wrap; }
        .header-title { display: flex; align-items: center; gap: 0.7rem; }
        .header-title svg { width: 30px; height: 30px; color: #818CF8; }
        .app-header h1 { color: #F8FAFC; font-size: 1.65rem; font-weight: 700; margin: 0; letter-spacing: -0.01em; }
        .app-header p { color: #94A3C4; font-size: 0.93rem; margin: 0.5rem 0 0 2.5rem; max-width: 34rem; }
        .status-pill {
            display: inline-flex; align-items: center; gap: 0.4rem;
            background: rgba(52, 211, 153, 0.12); border: 1px solid rgba(52, 211, 153, 0.35);
            color: #6EE7B7; font-size: 0.78rem; font-family: 'JetBrains Mono', monospace;
            padding: 0.35rem 0.7rem; border-radius: 999px;
        }
        .status-pill svg { width: 8px; height: 8px; }

        /* ---------- console groups (left column) ---------- */
        .console {
            background: #FFFFFF;
            border: 1px solid #E4E8F1;
            border-radius: 18px;
            padding: 1.5rem 1.6rem 0.4rem 1.6rem;
            box-shadow: 0 2px 14px rgba(15, 23, 42, 0.04);
        }
        .console-title { font-size: 1.05rem; font-weight: 700; color: #0F172A; margin-bottom: 0.15rem; }
        .console-sub { font-size: 0.83rem; color: #8792A8; margin-bottom: 1.1rem; }
        .group-label {
            display: flex; align-items: center; gap: 0.45rem;
            font-size: 0.86rem; font-weight: 600; color: #3730A3;
            margin: 1.1rem 0 0.6rem 0; padding-left: 0.6rem;
            border-left: 3px solid #4F46E5;
        }
        .group-label svg { width: 15px; height: 15px; color: #4F46E5; }

        /* Streamlit widget refinements */
        div[data-baseweb="select"] > div, .stNumberInput input {
            border-radius: 10px !important;
            border-color: #DDE2EE !important;
        }
        .stSlider { padding-top: 0.2rem; }
        label { font-size: 0.85rem !important; color: #3B4254 !important; font-weight: 500 !important; }

        div.stButton > button {
            background: #0F172A;
            color: #F8FAFC;
            font-weight: 600;
            font-family: 'Space Grotesk', sans-serif;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 1rem;
            width: 100%;
            font-size: 0.98rem;
            margin-top: 0.9rem;
            transition: transform 0.12s ease, background 0.12s ease;
        }
        div.stButton > button:hover { background: #1E293B; transform: translateY(-1px); }

        /* ---------- result panel (right column) ---------- */
        .result-panel {
            background: #FFFFFF;
            border: 1px solid #E4E8F1;
            border-radius: 18px;
            padding: 1.6rem;
            box-shadow: 0 2px 14px rgba(15, 23, 42, 0.04);
            min-height: 100%;
        }
        .panel-title { font-size: 1.05rem; font-weight: 700; color: #0F172A; margin-bottom: 1rem; }

        .idle-box {
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            text-align: center; padding: 2.6rem 1rem; border: 1.5px dashed #D8DEEB; border-radius: 14px;
            color: #9AA3B8;
        }
        .idle-box svg { width: 34px; height: 34px; color: #B7BFD4; margin-bottom: 0.7rem; }
        .idle-box .idle-title { font-family: 'Space Grotesk', sans-serif; font-weight: 600; color: #5B6478; font-size: 0.95rem; }
        .idle-box .idle-sub { font-size: 0.82rem; margin-top: 0.25rem; }

        .gauge-wrap { display: flex; flex-direction: column; align-items: center; margin-bottom: 0.4rem; }
        .gauge-value { font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 600; margin-top: -3.6rem; }
        .gauge-caption { font-size: 0.78rem; color: #8792A8; margin-top: 0.1rem; margin-bottom: 0.6rem; }

        .verdict-badge {
            display: flex; align-items: center; gap: 0.65rem;
            border-radius: 14px; padding: 0.9rem 1.1rem; margin-top: 0.4rem;
        }
        .verdict-fraud { background: #FEF2F2; border: 1px solid #FCA5A5; }
        .verdict-legit { background: #F0FDF6; border: 1px solid #86EFAC; }
        .verdict-badge svg { width: 26px; height: 26px; flex-shrink: 0; }
        .verdict-fraud svg { color: #DC2626; }
        .verdict-legit svg { color: #16A34A; }
        .verdict-badge h2 { margin: 0; font-size: 1.12rem; font-weight: 700; }
        .verdict-fraud h2 { color: #B91C1C; }
        .verdict-legit h2 { color: #15803D; }
        .verdict-badge p { margin: 0.1rem 0 0 0; font-size: 0.82rem; color: #6B7280; }

        .snapshot { margin-top: 1.3rem; }
        .snapshot-title { font-size: 0.8rem; color: #8792A8; margin-bottom: 0.5rem; font-weight: 500; }
        .snapshot-row {
            display: flex; justify-content: space-between; align-items: center;
            padding: 0.42rem 0.05rem; border-bottom: 1px dashed #EBEEF5;
            font-size: 0.83rem;
        }
        .snapshot-row:last-child { border-bottom: none; }
        .snapshot-key { color: #6B7280; display: flex; align-items: center; gap: 0.45rem; }
        .snapshot-key svg { width: 14px; height: 14px; color: #9AA3B8; }
        .snapshot-val { font-family: 'JetBrains Mono', monospace; color: #0F172A; font-weight: 500; }

        .footer-note { text-align: center; color: #A2AABD; font-size: 0.78rem; margin-top: 1.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 4. UI - HEADER
# =========================================================
st.markdown(
    f"""
    <div class="app-header">
        <div class="header-row">
            <div>
                <div class="header-title">{icon("radar", 30)}<h1>Credit Card Fraud Detection</h1></div>
                <p>Score a single transaction against a trained XGBoost pipeline and see the model's verdict in real time.</p>
            </div>
            <div class="status-pill">{icon("dot", 8)} model ready &middot; XGBoost pipeline</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Stop early with a clear error if the model could not be loaded
if load_error:
    st.error(f"{load_error}\n\nMake sure 'fraud_detection_model.pkl' is placed in the same folder as app.py.")
    st.stop()


# =========================================================
# 5. USER INPUT COLLECTION
# =========================================================
left_col, right_col = st.columns([1.15, 0.85], gap="large")

with left_col:
    st.markdown('<div class="console">', unsafe_allow_html=True)
    st.markdown('<div class="console-title">Transaction console</div>', unsafe_allow_html=True)
    st.markdown('<div class="console-sub">Fill in the fields below exactly as they would appear on an incoming transaction.</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="group-label">{icon("amount", 15)}Transaction</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        amount = st.number_input(
            "Amount ($)", min_value=0.0, max_value=100000.0, value=100.0, step=1.0,
            help="Total transaction amount in dollars.",
        )
    with c2:
        transaction_hour = st.slider(
            "Hour of day (0-23)", min_value=0, max_value=23, value=12,
            help="Hour the transaction occurred, 24-hour format.",
        )

    st.markdown(f'<div class="group-label">{icon("store", 15)}Context</div>', unsafe_allow_html=True)
    c3, c4, c5 = st.columns(3)
    with c3:
        merchant_category = st.selectbox(
            "Merchant category", options=["Electronics", "Travel", "Grocery", "Food", "Clothing"],
        )
    with c4:
        foreign_transaction_label = st.selectbox(
            "Foreign transaction", options=["No", "Yes"],
            help="Was this made in a foreign country?",
        )
    with c5:
        location_mismatch_label = st.selectbox(
            "Location mismatch", options=["No", "Yes"],
            help="Does the location differ from the cardholder's usual location?",
        )

    st.markdown(f'<div class="group-label">{icon("signal", 15)}Behaviour signals</div>', unsafe_allow_html=True)
    c6, c7 = st.columns(2)
    with c6:
        device_trust_score = st.slider(
            "Device trust score (0-100)", min_value=0, max_value=100, value=65,
            help="Trust score of the device used for the transaction.",
        )
    with c7:
        velocity_last_24h = st.number_input(
            "Velocity, last 24h", min_value=0, max_value=100, value=2, step=1,
            help="Number of transactions on this card in the last 24 hours.",
        )

    predict_clicked = st.button("Predict Fraud", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Convert Yes/No labels back to the 0/1 encoding used during training
foreign_transaction = 1 if foreign_transaction_label == "Yes" else 0
location_mismatch = 1 if location_mismatch_label == "Yes" else 0


# =========================================================
# 6. PREDICTION
# =========================================================
def render_gauge(pct: float) -> str:
    """Build an SVG semicircle risk gauge for the given probability (0-1)."""
    r = 80
    arc_len = math.pi * r
    offset = arc_len * (1 - max(0.0, min(1.0, pct)))
    color = "#16A34A" if pct < 0.4 else ("#F59E0B" if pct < 0.7 else "#DC2626")
    return f"""
    <div class="gauge-wrap">
        <svg viewBox="0 0 200 110" width="230" height="126">
            <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#E7EAF3" stroke-width="16" stroke-linecap="round"/>
            <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="{color}" stroke-width="16"
                  stroke-linecap="round" stroke-dasharray="{arc_len:.2f} {arc_len:.2f}" stroke-dashoffset="{offset:.2f}"/>
        </svg>
        <div class="gauge-value" style="color:{color};">{pct * 100:.1f}%</div>
        <div class="gauge-caption">estimated fraud probability</div>
    </div>
    """


with right_col:
    st.markdown('<div class="result-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Model verdict</div>', unsafe_allow_html=True)

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
                    fraud_probability = float(proba[1])  # class "1" = fraud
                except Exception:
                    fraud_probability = None

            if fraud_probability is not None:
                st.markdown(render_gauge(fraud_probability), unsafe_allow_html=True)

            if int(prediction) == 1:
                st.markdown(
                    f"""
                    <div class="verdict-badge verdict-fraud">
                        {icon("alert", 26)}
                        <div><h2>Fraudulent transaction</h2><p>Flagged as high-risk by the model.</p></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="verdict-badge verdict-legit">
                        {icon("shield", 26)}
                        <div><h2>Legitimate transaction</h2><p>No fraud signals detected.</p></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # transaction snapshot
            snapshot_rows = [
                ("amount", "Amount", f"${amount:,.2f}"),
                ("clock", "Hour", f"{transaction_hour:02d}:00"),
                ("store", "Merchant", merchant_category),
                ("globe", "Foreign txn", foreign_transaction_label),
                ("pin", "Location mismatch", location_mismatch_label),
                ("signal", "Device trust", f"{device_trust_score}/100"),
                ("repeat", "Velocity 24h", str(velocity_last_24h)),
            ]
            rows_html = "".join(
                f'<div class="snapshot-row"><span class="snapshot-key">{icon(ic, 14)}{label}</span>'
                f'<span class="snapshot-val">{val}</span></div>'
                for ic, label, val in snapshot_rows
            )
            st.markdown(
                f'<div class="snapshot"><div class="snapshot-title">Transaction snapshot</div>{rows_html}</div>',
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.markdown(
            f"""
            <div class="idle-box">
                {icon("radar", 34)}
                <div class="idle-title">Awaiting transaction data</div>
                <div class="idle-sub">Fill in the console on the left and click "Predict Fraud".</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer-note">Model pipeline: ColumnTransformer (StandardScaler + OneHotEncoder) &rarr; XGBoost Classifier</div>',
    unsafe_allow_html=True,
)
