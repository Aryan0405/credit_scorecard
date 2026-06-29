# Model Card: Retail Credit Risk Scorecard

---

## 1. Model Metadata & Overview

| Field | Details |
|---|---|
| **Model Name** | Home Credit Default Risk Scorecard |
| **Model Type** | Calibrated Logistic Regression (Production) & XGBoost (Challenger) |
| **Task** | Binary Classification (Predicting likelihood of credit default) |
| **Deployment Status** | Logistic Regression recommended for deployment; XGBoost-B rejected for production |
| **Version** | 1.0 | 
| **Date** | 29-06-2026 |
---

## 2. Intended Use & Out-of-Scope Uses

**Intended Use:** To rank-order retail credit applicants by default risk and assign interpretable credit scores. Designed to be used by credit officers as a primary decision-support tool.

**Intended Users:** Credit risk underwriters, loan origination systems (LOS), and model risk governance teams.

**Out-of-Scope Uses:**
- **Fully automated rejections without manual review:** The model is not designed to be the sole arbiter for credit denial, especially near the decision cutoff.
- **XGBoost-B for Production:** Rejected under RBI model risk guidelines. The marginal +0.0106 AUC gain does not justify the added model complexity, the architectural dependency on a secondary data source (Bureau), and the operational overhead of maintaining a calibrated Platt-scaling pipeline.

---

## 3. Data & Predictor Variables

- **Data Source:** Kaggle Home Credit Default Risk dataset (Application and Bureau data)
- **Target Variable:** `TARGET` (1 = Default, 0 = Non-Default)
- **Class Imbalance:** The training dataset exhibits a significant class imbalance with an approximate 8% default rate. This structural characteristic necessitated specific downstream modeling decisions (e.g., balanced class weighting and post-hoc intercept calibration).
- **Training/Test Split:** 80/20 split based on `DAYS_ID_PUBLISH`

**Selected Features (Logistic Baseline):**

| Feature | Description |
|---|---|
| `DAYS_BIRTH` | Age |
| `DAYS_EMPLOYED` | Tenure |
| `EXT_SOURCE_1/2/3` | Normalized scores derived from external data sources (effectively proxying external credit bureau history). These are the dominant predictive signals in the model. |

**Data Limitations:** The `DAYS_ID_PUBLISH` split is a proxy, not a genuine calendar-based temporal split, as the dataset lacks absolute datetime fields.

---

## 4. Methodology & Scoring Engine

**Algorithm Choice:** Logistic Regression on Weight of Evidence (WoE) transformed variables.

**Key Modeling Decisions:**

- **Feature Transformation:** Variables were discretized and transformed using OptimalBinning to handle non-linearities and missing values monotonically.
- **Imbalance Handling:** `class_weight='balanced'` was applied during training to improve minority class learning.
- **Probability Calibration:** A closed-form intercept correction was applied to correct the probability distortion caused by class balancing. Pre-fix mean predicted probability: 0.4833; Post-fix: 0.0949 (aligned with true test rate of 0.0999).

---

## 5. Model Performance & Validation

| Metric | Logistic Scorecard | XGBoost-A (5 raw features) | XGBoost-B (12 features) |
|---|---|---|---|
| ROC AUC | 0.7188 | 0.7282 | 0.7294 |
| KS Statistic | 0.3292 | 0.3395 | 0.3435 |

**Train/Test Stability:** The default rate increases from 7.59% in the training data to 9.99% in the test data. Because we don't have real dates to do a proper timeline split, we cannot prove the model handles future changes in the applicant pool well. This requires close monitoring once deployed.

---

## 6. Limitations & Known Issues

> These limitations must be reviewed before any production deployment.

- **WoE Binning Ceiling (Logistic Model):** The model cannot distinguish between extreme applicants who fall into identical worst-risk bins (e.g., `EXT_SOURCE_2_WoE = -1.143`). This results in identical penalty contributions for all extreme outliers, fundamentally limiting the model's ability to rank-differentiate within the highest-risk segment.

- **EXT_SOURCE_2 Missing Value Inconsistency:** In the XGBoost model, missing values (`NaN`) are routed inconsistently through default decision branches, resulting in vertical stacks in the SHAP dependence plot where some missing values receive a positive penalty and others a negative one.

- **Pseudo-Temporal Split:** The model relies on `DAYS_ID_PUBLISH` for train/test splitting, which does not guarantee protection against chronological data drift.

- **Bureau Feature Redundancy:** Adding explicit bureau variables (XGBoost-B) provided almost no lift over application variables. `EXT_SOURCE_2` and `EXT_SOURCE_3` already act as near-perfect proxies for external bureau risk.

---

## 7. Fairness & Explainability

**Explainability (Limited Scope):** The Logistic Regression baseline is globally and locally explained via a deployed Points Table (`outputs/scorecard_points_table.csv`). The XGBoost challenger was explained locally using SHAP TreeExplainer on top-risk applicants.

*Limitation: Comprehensive explanation coverage, including SHAP stability testing across score bands and edge cases, has not been conducted.*

---

>  **Fairness (OUT OF SCOPE)**
>
> No quantitative fairness or bias analysis has been conducted on this model. The model heavily utilizes `DAYS_BIRTH` (Age) and `DAYS_EMPLOYED` (Tenure/Employment Status). In highly regulated environments, these are known proxies for protected classes. **This model must undergo disparate impact testing across protected demographic cohorts before production deployment.**