# Credit Risk Scorecard — Home Credit Default Risk

A production-grade credit risk model built on the [Kaggle Home Credit Default Risk dataset](https://www.kaggle.com/c/home-credit-default-risk), comparing a regulatory-compliant Logistic Regression scorecard against an XGBoost challenger across model performance, explainability, and deployment suitability.

---

## Why It Matters

A naive approach would have just chased AUC and blindly deployed the highest-scoring model. Instead, we voluntarily rejected the XGBoost challenger—despite its +0.0106 higher AUC—because that marginal gain could not justify the regulatory complexity and operational overhead of maintaining a calibrated, secondary-data pipeline. Beyond just comparing models, we implemented SHAP for local and global explainability and prevented look-ahead data leakage by explicitly filtering out future-dated credit records.

---

## What Was Built

- **Production Scorecard (Logistic Regression):** A baseline model built on 5 Weight of Evidence (WoE) transformed variables, achieving a 0.7188 AUC.
- **Challenger Variant A (XGBoost):** A tree-based model built on the identical 5 raw features to isolate the pure algorithmic lift, achieving a 0.7282 AUC.
- **Challenger Variant B (XGBoost):** A calibrated tree-based model built on an expanded set of 12 raw features (including external bureau indicators), achieving a 0.7294 AUC.

---

## Key Findings

- **The WoE Ceiling:** Binning continuous variables into risk buckets fundamentally blinds a logistic model to extreme outliers. For the top three highest-risk applicants, the logistic model capped their default probability around 55% because they all hit the exact same "worst-case" WoE bin. The XGBoost model, operating on raw continuous features, correctly flagged those exact same individuals at ~92% risk.

- **Bureau Feature Redundancy:** Adding 7 explicit external credit bureau features (loan counts, overdue balances) provided negligible marginal value — only a +0.0012 AUC lift over the application-only model. SHAP analysis proved why: `EXT_SOURCE_2` and `EXT_SOURCE_3` alone drive over 60% of the model's decisions — they are already near-perfect proxies for bureau risk.

- **Algorithmic Lift vs. Information Lift:** Moving from a linear model to a non-linear tree algorithm (XGBoost-A) yielded a tangible +0.0094 AUC gain using the exact same 5 features. Moving from 5 features to 12 features (XGBoost-B) yielded almost nothing. The bottleneck in this dataset is the information itself, not the complexity of the math used to process it.

---

## Technical Stack

**Language:** Python 3.10

**Libraries:** NumPy, Pandas, SciPy, scikit-learn, XGBoost, SHAP, optbinning, joblib, matplotlib

---

## Repo Structure

```
credit_scorecard/
├── data/
│   └── raw/
│       ├── application_train.csv
│       └── bureau.csv
├── models/                        # Serialized model artifacts (joblib)
├── notebooks/
│   ├── 01_load_and_inspect_data.ipynb
│   └── shap_analysis.ipynb        # SHAP explainability analysis
├── outputs/
│   ├── scorecard_points_table.csv # Scorecard points table (PDO=20)
│   └── scorecard_results.csv
├── src/
│   ├── scorecard_pipeline.py      # Production LR scorecard
│   └── xgboost_challenger.py      # XGBoost challenger models
├── model_card.md                  # Full technical model card
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
git clone https://github.com/Aryan0405/credit_scorecard.git
cd credit_scorecard
conda create -n credit_scorecard python=3.10
conda activate credit_scorecard
pip install -r requirements.txt
```

Then open `notebooks/shap_analysis.ipynb` with the `credit_scorecard` kernel.

> **Note:** `xgboost==2.1.0` is pinned in `requirements.txt`. Do not upgrade — SHAP 0.49.1 is incompatible with XGBoost 3.x booster serialization.

> **Data:** Download `application_train.csv` and `bureau.csv` from the [Kaggle competition page](https://www.kaggle.com/c/home-credit-default-risk/data) and place them in `data/raw/`.

---

## Model Card

For full technical detail — methodology, validation approach, limitations, and fairness disclosures — see [`model_card.md`](./model_card.md).