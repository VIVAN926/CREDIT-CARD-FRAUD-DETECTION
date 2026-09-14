# Credit Card Fraud Detection

A machine learning project for detecting potentially fraudulent credit
card transactions using transaction-level behavioral and contextual
features.

## Overview

This project compares **Random Forest** and **XGBoost** classifiers for
binary fraud detection.

The workflow includes:

-   Data cleaning and preprocessing
-   Numerical feature scaling
-   Categorical feature encoding
-   Class-imbalance handling
-   Model training and evaluation
-   Saving the trained XGBoost pipeline with Joblib

## Dataset & Features

The dataset contains **10,000 transactions**, with fraud representing a
relatively small portion of the data.

### Features Used

  -----------------------------------------------------------------------
  Feature                             Description
  ----------------------------------- -----------------------------------
  `amount`                            Transaction amount

  `transaction_hour`                  Hour at which the transaction
                                      occurred

  `merchant_category`                 Merchant category

  `foreign_transaction`               Whether the transaction was made
                                      abroad

  `location_mismatch`                 Whether transaction location
                                      differs from the expected location

  `device_trust_score`                Trust score associated with the
                                      device

  `velocity_last_24h`                 Number of recent transactions
  -----------------------------------------------------------------------

`transaction_id` and `cardholder_age` were not used for model training.

## Pipeline Architecture

``` text
Raw Transaction Data
        │
        ▼
Feature Selection
        │
        ├── Numerical Features ──► StandardScaler
        │
        └── Categorical Features ─► OneHotEncoder
                         │
                         ▼
                  ColumnTransformer
                         │
                         ▼
                    Classifier
                         │
                         ▼
                  Fraud Prediction
```

The preprocessing and model are combined into a pipeline so that the
same transformations are applied during inference.

## Model Comparison

  Model             Accuracy   Precision   Recall   F1 Score
  --------------- ---------- ----------- -------- ----------
  Random Forest       99.80%        100%   86.67%     92.86%
  XGBoost               100%        100%     100%       100%

These are the **test-set results recorded in the current notebook**, not
estimated or adjusted values.

The XGBoost result is unusually strong. It should therefore be
interpreted cautiously rather than presented as evidence that the model
would achieve 100% performance on real-world banking data. The dataset
is relatively small and comes from Kaggle.

## Project Structure

``` text
credit-card-fraud-detection/
│
├── credit_card_fraud_10k.csv
├── fraud_detection_model.pkl
├── PROJ2(1).ipynb
├── requirements.txt
├── README.md
│
└── app/
    ├── app.py
    ├── templates/
    │   └── index.html
    └── static/
        ├── style.css
        └── script.js
```

> Adjust the structure above to match the actual files in the
> repository.

## Quickstart

### Installation

``` bash
git clone <your-repository-url>
cd credit-card-fraud-detection
pip install -r requirements.txt
```

### Usage

The trained pipeline is stored as `fraud_detection_model.pkl`.

``` python
import joblib
import pandas as pd
model = joblib.load("fraud_detection_model.pkl")
sample = pd.DataFrame([{"amount": 150, "transaction_hour": 22, "merchant_category": "online", "foreign_transaction": 1, "location_mismatch": 1, "device_trust_score": 35, "velocity_last_24h": 5}])
print("Fraudulent" if model.predict(sample)[0] == 1 else "Legitimate")
```

## Tech Stack

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   XGBoost
-   Matplotlib / Seaborn
-   Joblib

## Limitations

This is an **educational/portfolio project** built using a Kaggle
dataset. The reported test performance should not be interpreted as
production-level fraud detection performance.

## Author

**Vivan**

If you found the project useful, feel free to explore the notebook and
implementation.
