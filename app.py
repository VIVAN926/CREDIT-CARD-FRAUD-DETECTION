"""
app.py — Integration / deployment layer for the Credit Card Fraud Detection project.

This file does NOT contain any ML logic. It exists purely to:
  1. Load the existing, untouched trained pipeline from protected/fraud_detection_model.pkl
  2. Accept form input from the web UI
  3. Validate + shape that input into a pandas DataFrame with the exact column
     names the pipeline's ColumnTransformer was fit on
  4. Call the pipeline's own .predict() / .predict_proba() methods
  5. Return the model's real output as JSON

No preprocessing, encoding, or scaling logic is reimplemented here — the
joblib file is a full sklearn Pipeline (ColumnTransformer + XGBClassifier),
so calling .predict() on it runs the original preprocessing exactly as
trained in PROJ2.ipynb.
"""

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "protected", "fraud_detection_model.pkl")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the existing trained pipeline exactly as-is. We never retrain,
# refit, or alter this object in any way.
# ---------------------------------------------------------------------------
model = joblib.load(MODEL_PATH)

# The exact feature columns the ColumnTransformer inside the pipeline expects
# (taken directly from PROJ2.ipynb — x = df.drop(["is_fraud"], axis=1) after
# transaction_id and cardholder_age were dropped). Order does not matter to
# a ColumnTransformer since it selects by name, but we keep this list for
# validation and to build the DataFrame with the right dtypes/columns.
NUMERIC_FEATURES = ["amount", "transaction_hour", "device_trust_score", "velocity_last_24h"]
CATEGORICAL_FEATURES = ["merchant_category", "location_mismatch", "foreign_transaction"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

VALID_MERCHANT_CATEGORIES = {"Electronics", "Travel", "Grocery", "Food", "Clothing"}

# Reasonable input bounds for validation/UI only — these do NOT feed into the
# model's math (the pipeline's own StandardScaler/OneHotEncoder handle that).
FIELD_RULES = {
    "amount": {"type": float, "min": 0, "max": 1_000_000},
    "transaction_hour": {"type": int, "min": 0, "max": 23},
    "device_trust_score": {"type": int, "min": 0, "max": 100},
    "velocity_last_24h": {"type": int, "min": 0, "max": 100},
    "foreign_transaction": {"type": int, "min": 0, "max": 1},
    "location_mismatch": {"type": int, "min": 0, "max": 1},
}

HAS_PREDICT_PROBA = hasattr(model, "predict_proba")


def validate_and_coerce(payload: dict):
    """Validate raw JSON input and coerce it into correct types.
    Returns (clean_dict, errors_dict)."""
    errors = {}
    clean = {}

    for field in ALL_FEATURES:
        if field not in payload or payload[field] in (None, ""):
            errors[field] = "This field is required."
            continue

        if field == "merchant_category":
            value = str(payload[field]).strip()
            if value not in VALID_MERCHANT_CATEGORIES:
                errors[field] = f"Must be one of: {', '.join(sorted(VALID_MERCHANT_CATEGORIES))}"
            else:
                clean[field] = value
            continue

        rule = FIELD_RULES[field]
        raw = payload[field]
        try:
            value = rule["type"](raw)
        except (TypeError, ValueError):
            errors[field] = "Must be a number."
            continue

        if value < rule["min"] or value > rule["max"]:
            errors[field] = f"Must be between {rule['min']} and {rule['max']}."
            continue

        clean[field] = value

    return clean, errors


@app.route("/")
def index():
    return render_template(
        "index.html",
        merchant_categories=sorted(VALID_MERCHANT_CATEGORIES),
        has_proba=HAS_PREDICT_PROBA,
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    clean, errors = validate_and_coerce(payload)

    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    # Build a single-row DataFrame with the exact column names the pipeline
    # was fit on. Column order is irrelevant to ColumnTransformer (it selects
    # by name), but we pass it in the same order as training for clarity.
    row = pd.DataFrame([{col: clean[col] for col in ALL_FEATURES}])

    # --- Call the existing model's own prediction methods. No custom logic. ---
    prediction = model.predict(row)[0]
    result = {
        "ok": True,
        "prediction": int(prediction),
        "label": "Fraudulent" if int(prediction) == 1 else "Legitimate",
    }

    if HAS_PREDICT_PROBA:
        proba = model.predict_proba(row)[0]
        result["probability_legitimate"] = float(proba[0])
        result["probability_fraud"] = float(proba[1])

    return jsonify(result)


@app.route("/api/example", methods=["GET"])
def example():
    """Returns a random real row from the original dataset (features only)
    so the UI's 'Try an example' button can populate the form with genuine
    data rather than made-up numbers."""
    csv_path = os.path.join(BASE_DIR, "protected", "credit_card_fraud_10k.csv")
    df = pd.read_csv(csv_path)
    sample = df.sample(1).iloc[0]
    return jsonify({
        "amount": round(float(sample["amount"]), 2),
        "transaction_hour": int(sample["transaction_hour"]),
        "merchant_category": sample["merchant_category"],
        "foreign_transaction": int(sample["foreign_transaction"]),
        "location_mismatch": int(sample["location_mismatch"]),
        "device_trust_score": int(sample["device_trust_score"]),
        "velocity_last_24h": int(sample["velocity_last_24h"]),
        "actual_label": int(sample["is_fraud"]),
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
