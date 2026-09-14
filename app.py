"""
app.py — Single-file web app for the Credit Card Fraud Detection project.

Everything for the web layer (HTML, CSS, JS) lives in this one file — no
separate templates/ or static/ folders are needed. The ML pipeline itself is
NOT reimplemented anywhere below: it is loaded from your existing
fraud_detection_model.pkl exactly as you saved it with joblib, and
.predict() / .predict_proba() are called directly on that object.

Your original ML files are treated as read-only:
  - fraud_detection_model.pkl  (loaded with joblib.load, never modified)
  - credit_card_fraud_10k.csv  (only read, to power the optional
                                 "Load example" button — the app still
                                 works fully without it)
  - PROJ2.ipynb                (kept for reference, not executed here)

Model path resolution
----------------------
This looks for the model beside app.py first (flat layout):

    PROJECT 2/
    ├── app.py
    ├── fraud_detection_model.pkl
    ├── requirements.txt
    └── README.md

and falls back to a protected/ subfolder if that's how your files are
organized instead. Either layout works without editing this file.
"""

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_path(filename: str) -> str:
    """Find `filename` beside app.py, or inside a protected/ subfolder.
    Raises a clear error if it can't be found in either place."""
    flat_path = os.path.join(BASE_DIR, filename)
    nested_path = os.path.join(BASE_DIR, "protected", filename)

    if os.path.exists(flat_path):
        return flat_path
    if os.path.exists(nested_path):
        return nested_path

    raise FileNotFoundError(
        f"Could not find '{filename}'. Expected it either at:\n"
        f"  {flat_path}\n"
        f"or:\n"
        f"  {nested_path}"
    )


MODEL_PATH = _resolve_path("fraud_detection_model.pkl")

# The CSV is optional — only the "Load example" button needs it. The app
# still runs and makes real predictions without it.
try:
    CSV_PATH = _resolve_path("credit_card_fraud_10k.csv")
except FileNotFoundError:
    CSV_PATH = None

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the existing trained pipeline exactly as-is. Never retrained,
# refit, or altered here.
# ---------------------------------------------------------------------------
model = joblib.load(MODEL_PATH)

NUMERIC_FEATURES = ["amount", "transaction_hour", "device_trust_score", "velocity_last_24h"]
CATEGORICAL_FEATURES = ["merchant_category", "location_mismatch", "foreign_transaction"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

VALID_MERCHANT_CATEGORIES = {"Electronics", "Travel", "Grocery", "Food", "Clothing"}

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


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Transaction Risk Console — Fraud Detection</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0a0f1a;
  --bg-grid: #0d1420;
  --surface: #121a28;
  --surface-2: #182338;
  --border: #253449;
  --border-soft: #1c2a3f;
  --text: #e9eef7;
  --text-muted: #8a97b3;
  --text-faint: #5c6a86;
  --amber: #e8a33d;
  --amber-soft: rgba(232, 163, 61, 0.14);
  --safe: #35c48c;
  --safe-soft: rgba(53, 196, 140, 0.14);
  --danger: #ef5b5b;
  --danger-soft: rgba(239, 91, 91, 0.14);
  --radius: 14px;
  --font-display: "Space Grotesk", sans-serif;
  --font-mono: "IBM Plex Mono", monospace;
}

* { box-sizing: border-box; }

html, body {
  margin: 0;
  padding: 0;
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-display);
  -webkit-font-smoothing: antialiased;
}

body {
  background-image:
    linear-gradient(var(--bg-grid) 1px, transparent 1px),
    linear-gradient(90deg, var(--bg-grid) 1px, transparent 1px);
  background-size: 42px 42px;
  background-position: center top;
}

.scanline {
  position: fixed;
  inset: 0;
  pointer-events: none;
  background: radial-gradient(ellipse 80% 50% at 50% -10%, rgba(232,163,61,0.06), transparent 60%);
  z-index: 0;
}

a { color: inherit; }

/* ---------- Topbar ---------- */
.topbar {
  position: sticky;
  top: 0;
  z-index: 10;
  background: rgba(10, 15, 26, 0.85);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--border-soft);
}
.topbar__inner {
  max-width: 1100px;
  margin: 0 auto;
  padding: 16px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.brand { display: flex; align-items: center; gap: 10px; font-weight: 600; letter-spacing: 0.01em; }
.brand__mark { color: var(--amber); font-size: 1.1rem; }
.topbar__meta {
  display: flex;
  align-items: center;
  gap: 8px;
  font-family: var(--font-mono);
  font-size: 0.78rem;
  color: var(--text-muted);
}
.dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
.dot--live { background: var(--safe); box-shadow: 0 0 0 3px var(--safe-soft); }

/* ---------- Layout ---------- */
.layout {
  max-width: 1100px;
  margin: 0 auto;
  padding: 56px 24px 80px;
  position: relative;
  z-index: 1;
}

.hero { max-width: 640px; margin-bottom: 40px; }
.hero h1 {
  font-size: clamp(2rem, 4vw, 2.6rem);
  line-height: 1.15;
  margin: 0 0 16px;
  font-weight: 700;
  letter-spacing: -0.01em;
}
.hero__sub {
  color: var(--text-muted);
  font-size: 1.02rem;
  line-height: 1.6;
  margin: 0;
  max-width: 54ch;
}

/* ---------- Console grid ---------- */
.console {
  display: grid;
  grid-template-columns: 1.1fr 1fr;
  gap: 20px;
  align-items: start;
}
@media (max-width: 860px) {
  .console { grid-template-columns: 1fr; }
}

.panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 26px;
}
.panel__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 22px;
  gap: 12px;
  flex-wrap: wrap;
}
.panel__header h2 {
  font-family: var(--font-mono);
  font-size: 0.85rem;
  font-weight: 500;
  color: var(--text-muted);
  letter-spacing: 0.02em;
  margin: 0;
}
.panel__actions { display: flex; gap: 8px; }

/* ---------- Form ---------- */
.field { margin-bottom: 18px; }
.field-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
.field-row .field { margin-bottom: 18px; }
@media (max-width: 480px) {
  .field-row { grid-template-columns: 1fr; }
}

label {
  display: block;
  font-size: 0.86rem;
  color: var(--text-muted);
  margin-bottom: 8px;
}
.unit {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  color: var(--text-faint);
  margin-left: 4px;
}

input[type="number"],
select {
  width: 100%;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 9px;
  padding: 11px 13px;
  color: var(--text);
  font-family: var(--font-mono);
  font-size: 0.95rem;
  outline: none;
  transition: border-color 0.15s ease;
  appearance: none;
}
select {
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='10' height='6'><path d='M0 0l5 6 5-6z' fill='%238a97b3'/></svg>");
  background-repeat: no-repeat;
  background-position: right 14px center;
  font-family: var(--font-display);
}
input[type="number"]:focus,
select:focus {
  border-color: var(--amber);
}
input[type="number"]::-webkit-outer-spin-button,
input[type="number"]::-webkit-inner-spin-button {
  opacity: 1;
}

input[type="range"] {
  width: 100%;
  accent-color: var(--amber);
  height: 4px;
  margin-top: 6px;
}
.range__scale {
  display: flex;
  justify-content: space-between;
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: var(--text-faint);
  margin-top: 6px;
}

.field__error {
  color: var(--danger);
  font-size: 0.78rem;
  margin: 6px 0 0;
  min-height: 1em;
}

.toggles { display: flex; gap: 24px; align-items: center; }
.toggle {
  display: flex;
  align-items: center;
  gap: 10px;
  cursor: pointer;
  margin: 0;
}
.toggle input { position: absolute; opacity: 0; width: 0; height: 0; }
.toggle__box {
  width: 38px;
  height: 22px;
  border-radius: 12px;
  background: var(--surface-2);
  border: 1px solid var(--border);
  position: relative;
  transition: background 0.15s ease, border-color 0.15s ease;
  flex-shrink: 0;
}
.toggle__box::after {
  content: "";
  position: absolute;
  top: 2px;
  left: 2px;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--text-faint);
  transition: transform 0.15s ease, background 0.15s ease;
}
.toggle input:checked + .toggle__box {
  background: var(--amber-soft);
  border-color: var(--amber);
}
.toggle input:checked + .toggle__box::after {
  transform: translateX(16px);
  background: var(--amber);
}
.toggle__label { font-size: 0.88rem; color: var(--text); }

/* ---------- Buttons ---------- */
.btn {
  font-family: var(--font-display);
  font-size: 0.88rem;
  font-weight: 600;
  border-radius: 9px;
  border: 1px solid transparent;
  cursor: pointer;
  transition: all 0.15s ease;
}
.btn--ghost {
  background: transparent;
  border-color: var(--border);
  color: var(--text-muted);
  padding: 7px 13px;
  font-weight: 500;
}
.btn--ghost:hover { border-color: var(--amber); color: var(--text); }

.btn--primary {
  width: 100%;
  background: var(--amber);
  color: #1a1206;
  padding: 14px;
  margin-top: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  font-size: 0.95rem;
}
.btn--primary:hover { background: #f0b559; }
.btn--primary:disabled { opacity: 0.7; cursor: progress; }

.btn__spinner {
  width: 14px;
  height: 14px;
  border-radius: 50%;
  border: 2px solid rgba(26,18,6,0.35);
  border-top-color: #1a1206;
  display: none;
  animation: spin 0.7s linear infinite;
}
.btn--primary.is-loading .btn__spinner { display: inline-block; }
.btn--primary.is-loading .btn__label { opacity: 0.85; }
@keyframes spin { to { transform: rotate(360deg); } }

.form__note {
  min-height: 1.2em;
  font-size: 0.82rem;
  color: var(--danger);
  margin: 10px 0 0;
  text-align: center;
}

/* ---------- Result panel ---------- */
.panel--result { min-height: 380px; display: flex; flex-direction: column; }
.result {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  color: var(--text-faint);
  gap: 12px;
  padding: 20px 10px;
}
.result__icon { font-size: 1.6rem; color: var(--border); }
.result p { max-width: 30ch; font-size: 0.9rem; line-height: 1.5; margin: 0; }

.result--card { align-items: stretch; text-align: left; animation: fadeUp 0.35s ease; }
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}

.verdict {
  display: flex;
  align-items: center;
  gap: 20px;
  padding-bottom: 22px;
  border-bottom: 1px solid var(--border-soft);
  margin-bottom: 20px;
}
.verdict__ring {
  width: 84px;
  height: 84px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  font-family: var(--font-mono);
  font-weight: 600;
  font-size: 1.05rem;
  border: 3px solid var(--border);
  background: var(--surface-2);
}
.verdict__ring.is-safe { border-color: var(--safe); color: var(--safe); }
.verdict__ring.is-danger { border-color: var(--danger); color: var(--danger); }

.verdict__badge {
  display: inline-block;
  font-family: var(--font-mono);
  font-size: 0.75rem;
  letter-spacing: 0.03em;
  padding: 4px 10px;
  border-radius: 20px;
  margin-bottom: 8px;
}
.verdict__badge.is-safe { background: var(--safe-soft); color: var(--safe); }
.verdict__badge.is-danger { background: var(--danger-soft); color: var(--danger); }
.verdict__desc { margin: 0; color: var(--text-muted); font-size: 0.9rem; line-height: 1.5; }

.confidence { margin-bottom: 22px; }
.confidence__row {
  display: flex;
  justify-content: space-between;
  font-size: 0.82rem;
  color: var(--text-muted);
  margin-bottom: 6px;
}
.confidence__value { font-family: var(--font-mono); color: var(--text); }
.confidence__bar {
  height: 7px;
  border-radius: 4px;
  background: var(--surface-2);
  overflow: hidden;
  margin-bottom: 16px;
}
.confidence__fill { height: 100%; width: 0%; border-radius: 4px; transition: width 0.6s cubic-bezier(.4,0,.2,1); }
.confidence__fill--safe { background: var(--safe); }
.confidence__fill--danger { background: var(--danger); }

.summary {
  margin: 0;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px 18px;
  font-size: 0.82rem;
}
.summary dt { color: var(--text-faint); }
.summary dd { margin: 2px 0 0; font-family: var(--font-mono); color: var(--text); }

/* ---------- About ---------- */
.about { margin-top: 56px; }
.about h2 {
  font-size: 1.3rem;
  margin-bottom: 20px;
}
.about__grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
}
@media (max-width: 780px) {
  .about__grid { grid-template-columns: 1fr; }
}
.about__card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
}
.about__label {
  display: block;
  font-family: var(--font-mono);
  font-size: 0.72rem;
  color: var(--amber);
  letter-spacing: 0.03em;
  margin-bottom: 10px;
}
.about__card p { color: var(--text-muted); font-size: 0.88rem; line-height: 1.55; margin: 0; }
.about__card p.mono { font-family: var(--font-mono); font-size: 0.8rem; }
.about__footnote { margin-top: 12px; font-size: 0.76rem !important; color: var(--text-faint) !important; }

.metrics { list-style: none; margin: 0; padding: 0; }
.metrics li {
  display: flex;
  justify-content: space-between;
  padding: 7px 0;
  border-bottom: 1px solid var(--border-soft);
  font-size: 0.86rem;
}
.metrics li:last-child { border-bottom: none; }
.metrics li span { color: var(--text-muted); }
.metrics li strong { font-family: var(--font-mono); color: var(--safe); }

/* ---------- Footer ---------- */
.footer {
  margin-top: 48px;
  padding-top: 24px;
  border-top: 1px solid var(--border-soft);
  color: var(--text-faint);
  font-size: 0.8rem;
  line-height: 1.6;
}
.footer code {
  font-family: var(--font-mono);
  background: var(--surface-2);
  padding: 2px 6px;
  border-radius: 4px;
}

</style>
</head>
<body>

<div class="scanline" aria-hidden="true"></div>

<header class="topbar">
  <div class="topbar__inner">
    <div class="brand">
      <span class="brand__mark">◈</span>
      <span class="brand__text">Transaction Risk Console</span>
    </div>
    <div class="topbar__meta">
      <span class="dot dot--live"></span>
      <span>model online · xgboost pipeline</span>
    </div>
  </div>
</header>

<main class="layout">

  <section class="hero">
    <h1>Score a transaction<br>before it clears.</h1>
    <p class="hero__sub">
      Enter the details of a card transaction below. The console sends it straight to a trained
      classification pipeline — scaling, encoding and all — and returns a real verdict, not a guess.
    </p>
  </section>

  <div class="console">

    <!-- INPUT PANEL -->
    <section class="panel panel--form" aria-label="Transaction details">
      <div class="panel__header">
        <h2>01 · Transaction details</h2>
        <div class="panel__actions">
          <button type="button" class="btn btn--ghost" id="exampleBtn">Load example</button>
          <button type="button" class="btn btn--ghost" id="resetBtn">Reset</button>
        </div>
      </div>

      <form id="predictForm" novalidate>
        <div class="field">
          <label for="amount">Transaction amount <span class="unit">USD</span></label>
          <input type="number" id="amount" name="amount" step="0.01" min="0" placeholder="e.g. 84.47" required>
          <p class="field__error" data-error-for="amount"></p>
        </div>

        <div class="field-row">
          <div class="field">
            <label for="transaction_hour">Hour of transaction <span class="unit">0–23</span></label>
            <input type="number" id="transaction_hour" name="transaction_hour" min="0" max="23" step="1" placeholder="e.g. 22" required>
            <p class="field__error" data-error-for="transaction_hour"></p>
          </div>
          <div class="field">
            <label for="merchant_category">Merchant category</label>
            <select id="merchant_category" name="merchant_category" required>
              <option value="" disabled selected>Select category</option>
              {% for cat in merchant_categories %}
              <option value="{{ cat }}">{{ cat }}</option>
              {% endfor %}
            </select>
            <p class="field__error" data-error-for="merchant_category"></p>
          </div>
        </div>

        <div class="field">
          <label for="device_trust_score">Device trust score <span class="unit" id="dtsVal">—</span></label>
          <input type="range" id="device_trust_score" name="device_trust_score" min="0" max="100" step="1" value="60">
          <div class="range__scale"><span>Untrusted</span><span>Trusted</span></div>
          <p class="field__error" data-error-for="device_trust_score"></p>
        </div>

        <div class="field-row">
          <div class="field">
            <label for="velocity_last_24h">Transactions in last 24h</label>
            <input type="number" id="velocity_last_24h" name="velocity_last_24h" min="0" max="100" step="1" placeholder="e.g. 3" required>
            <p class="field__error" data-error-for="velocity_last_24h"></p>
          </div>
        </div>

        <div class="field-row toggles">
          <label class="toggle">
            <input type="checkbox" id="foreign_transaction" name="foreign_transaction">
            <span class="toggle__box"></span>
            <span class="toggle__label">Foreign transaction</span>
          </label>
          <label class="toggle">
            <input type="checkbox" id="location_mismatch" name="location_mismatch">
            <span class="toggle__box"></span>
            <span class="toggle__label">Location mismatch</span>
          </label>
        </div>

        <button type="submit" class="btn btn--primary" id="submitBtn">
          <span class="btn__label">Run fraud check</span>
          <span class="btn__spinner" aria-hidden="true"></span>
        </button>
        <p class="form__note" id="formNote"></p>
      </form>
    </section>

    <!-- RESULT PANEL -->
    <section class="panel panel--result" aria-label="Prediction result">
      <div class="panel__header">
        <h2>02 · Model verdict</h2>
      </div>

      <div class="result" id="resultEmpty">
        <div class="result__icon">◇</div>
        <p>Fill in the transaction details and run a check to see the model's verdict here.</p>
      </div>

      <div class="result result--card" id="resultCard" hidden>
        <div class="verdict" id="verdict">
          <div class="verdict__ring" id="verdictRing">
            <span class="verdict__ringlabel" id="verdictPct">—</span>
          </div>
          <div class="verdict__text">
            <span class="verdict__badge" id="verdictBadge">—</span>
            <p class="verdict__desc" id="verdictDesc">—</p>
          </div>
        </div>

        {% if has_proba %}
        <div class="confidence">
          <div class="confidence__row">
            <span>Legitimate</span>
            <span class="confidence__value" id="probLegit">—</span>
          </div>
          <div class="confidence__bar">
            <div class="confidence__fill confidence__fill--safe" id="probLegitBar"></div>
          </div>
          <div class="confidence__row">
            <span>Fraudulent</span>
            <span class="confidence__value" id="probFraud">—</span>
          </div>
          <div class="confidence__bar">
            <div class="confidence__fill confidence__fill--danger" id="probFraudBar"></div>
          </div>
        </div>
        {% endif %}

        <dl class="summary" id="summaryList"></dl>
      </div>
    </section>

  </div>

  <section class="about">
    <h2>About this model</h2>
    <div class="about__grid">
      <div class="about__card">
        <span class="about__label">Pipeline</span>
        <p>ColumnTransformer (StandardScaler + OneHotEncoder) feeding an XGBoost classifier, trained with class-weighted scaling to correct for a ~1.5% fraud rate.</p>
      </div>
      <div class="about__card">
        <span class="about__label">Held-out test metrics</span>
        <ul class="metrics">
          <li><span>Accuracy</span><strong>100%</strong></li>
          <li><span>Precision</span><strong>100%</strong></li>
          <li><span>Recall</span><strong>100%</strong></li>
          <li><span>F1</span><strong>100%</strong></li>
        </ul>
        <p class="about__footnote">Evaluated on the same stratified 80/20 split used during training (2,000 held-out transactions, 30 fraudulent).</p>
      </div>
      <div class="about__card">
        <span class="about__label">Features used</span>
        <p class="mono">amount · transaction_hour · merchant_category · foreign_transaction · location_mismatch · device_trust_score · velocity_last_24h</p>
      </div>
    </div>
  </section>

  <footer class="footer">
    <p>Every prediction shown here is produced by calling <code>.predict()</code> / <code>.predict_proba()</code> on the original trained pipeline — no simulated output.</p>
  </footer>

</main>

<script>
const form = document.getElementById("predictForm");
const submitBtn = document.getElementById("submitBtn");
const formNote = document.getElementById("formNote");
const resultEmpty = document.getElementById("resultEmpty");
const resultCard = document.getElementById("resultCard");
const dtsInput = document.getElementById("device_trust_score");
const dtsVal = document.getElementById("dtsVal");

function updateDtsLabel() {
  dtsVal.textContent = dtsInput.value;
}
dtsInput.addEventListener("input", updateDtsLabel);
updateDtsLabel();

function clearErrors() {
  document.querySelectorAll(".field__error").forEach(el => (el.textContent = ""));
  formNote.textContent = "";
}

function showErrors(errors) {
  Object.entries(errors).forEach(([field, message]) => {
    const el = document.querySelector(`[data-error-for="${field}"]`);
    if (el) el.textContent = message;
  });
}

function collectPayload() {
  const fd = new FormData(form);
  return {
    amount: fd.get("amount"),
    transaction_hour: fd.get("transaction_hour"),
    merchant_category: fd.get("merchant_category"),
    device_trust_score: dtsInput.value,
    velocity_last_24h: fd.get("velocity_last_24h"),
    foreign_transaction: document.getElementById("foreign_transaction").checked ? 1 : 0,
    location_mismatch: document.getElementById("location_mismatch").checked ? 1 : 0,
  };
}

function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  submitBtn.classList.toggle("is-loading", isLoading);
}

function renderResult(data, payload) {
  resultEmpty.hidden = true;
  resultCard.hidden = false;

  const isFraud = data.prediction === 1;
  const ring = document.getElementById("verdictRing");
  const badge = document.getElementById("verdictBadge");
  const desc = document.getElementById("verdictDesc");
  const pct = document.getElementById("verdictPct");

  ring.classList.remove("is-safe", "is-danger");
  badge.classList.remove("is-safe", "is-danger");

  if (isFraud) {
    ring.classList.add("is-danger");
    badge.classList.add("is-danger");
    badge.textContent = "FLAGGED";
    desc.textContent = "The model classifies this transaction as likely fraudulent. It would typically be held for review.";
  } else {
    ring.classList.add("is-safe");
    badge.classList.add("is-safe");
    badge.textContent = "CLEARED";
    desc.textContent = "The model classifies this transaction as legitimate based on the pattern of inputs provided.";
  }

  if (typeof data.probability_fraud === "number") {
    const fraudPct = Math.round(data.probability_fraud * 100);
    const legitPct = Math.round(data.probability_legitimate * 100);
    pct.textContent = `${isFraud ? fraudPct : legitPct}%`;

    document.getElementById("probLegit").textContent = `${legitPct}%`;
    document.getElementById("probFraud").textContent = `${fraudPct}%`;
    requestAnimationFrame(() => {
      document.getElementById("probLegitBar").style.width = `${legitPct}%`;
      document.getElementById("probFraudBar").style.width = `${fraudPct}%`;
    });
  } else {
    pct.textContent = data.label;
  }

  const summary = document.getElementById("summaryList");
  summary.innerHTML = "";
  const rows = [
    ["Amount", `$${Number(payload.amount).toFixed(2)}`],
    ["Hour", payload.transaction_hour],
    ["Category", payload.merchant_category],
    ["Device trust", payload.device_trust_score],
    ["24h velocity", payload.velocity_last_24h],
    ["Foreign", payload.foreign_transaction ? "Yes" : "No"],
    ["Loc. mismatch", payload.location_mismatch ? "Yes" : "No"],
  ];
  rows.forEach(([label, value]) => {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value;
    summary.appendChild(dt);
    summary.appendChild(dd);
  });
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearErrors();

  const payload = collectPayload();
  setLoading(true);

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      showErrors(data.errors || {});
      formNote.textContent = "Please fix the highlighted fields.";
      return;
    }

    renderResult(data, payload);
  } catch (err) {
    formNote.textContent = "Could not reach the prediction server. Is app.py running?";
  } finally {
    setLoading(false);
  }
});

document.getElementById("resetBtn").addEventListener("click", () => {
  form.reset();
  dtsInput.value = 60;
  updateDtsLabel();
  clearErrors();
  resultEmpty.hidden = false;
  resultCard.hidden = true;
});

document.getElementById("exampleBtn").addEventListener("click", async () => {
  try {
    const res = await fetch("/api/example");
    const data = await res.json();

    if (!res.ok) {
      formNote.textContent = data.error || "Could not load an example.";
      return;
    }

    document.getElementById("amount").value = data.amount;
    document.getElementById("transaction_hour").value = data.transaction_hour;
    document.getElementById("merchant_category").value = data.merchant_category;
    dtsInput.value = data.device_trust_score;
    updateDtsLabel();
    document.getElementById("velocity_last_24h").value = data.velocity_last_24h;
    document.getElementById("foreign_transaction").checked = !!data.foreign_transaction;
    document.getElementById("location_mismatch").checked = !!data.location_mismatch;

    clearErrors();
    formNote.textContent = `Loaded a real row from the dataset (actual label: ${data.actual_label === 1 ? "fraud" : "legitimate"}).`;
    formNote.style.color = "var(--text-muted)";
  } catch (err) {
    formNote.textContent = "Could not load an example.";
  }
});

</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(
        PAGE_TEMPLATE,
        merchant_categories=sorted(VALID_MERCHANT_CATEGORIES),
        has_proba=HAS_PREDICT_PROBA,
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    clean, errors = validate_and_coerce(payload)

    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    row = pd.DataFrame([{col: clean[col] for col in ALL_FEATURES}])

    # --- Existing model's own prediction methods. No custom logic. ---
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
    if CSV_PATH is None:
        return jsonify({
            "ok": False,
            "error": "credit_card_fraud_10k.csv was not found, so example data isn't available. "
                     "Predictions still work normally.",
        }), 404
    df = pd.read_csv(CSV_PATH)
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
