# Credit Risk Prediction Model: Home Credit Default Analysis

Credit risk is the risk that a borrower fails to repay what they owe. Banks, NBFCs, and fintech lenders quantify this risk for two major reasons: (1) Pricing - charging higher interest rates to riskier borrowers, and (2) Capital Adequacy - regulators under the Basel Committee on Banking Supervision framework require banks to hold capital buffers against potential losses.

The objective of this project was to predict the probability of applicants experiencing payment difficulties according to Home Credit internal criteria (TARGET = 1). The project implements two models: a production-grade logistic regression scorecard and an XGBoost challenger model, compared under a controlled experimental design.

---

## Repository Structure

| File | Role |
| :--- | :--- |
| `scorecard_pipeline.py` | Production scorecard - formally validated, regulator-facing |
| `xgboost_challenger.py` | Challenger model - experimental, not validated for deployment |

This separation is intentional and mirrors real bank model-risk practice: production scorecard variables go through a formal validation gate; challenger models use a broader, less-validated feature set for benchmarking purposes.

---

## Model 1: Logistic Regression Scorecard (`scorecard_pipeline.py`)

### Features
Rigorous feature selection was conducted using Information Value (IV) to prioritise predictive power while maintaining business interpretability.

| Feature | Information Value (IV) |
| :--- | :--- |
| EXT_SOURCE_3 | 0.3282 |
| EXT_SOURCE_2 | 0.3119 |
| EXT_SOURCE_1 | 0.1396 |
| DAYS_EMPLOYED | 0.1093 |
| DAYS_BIRTH | 0.0853 |

### Methodology
- Weight of Evidence (WoE) binning transforms variables into monotonic risk relationships suitable for logistic regression.
- Feature selection via IV and business interpretability avoids unstable predictors and multicollinearity.
- Proxy temporal split using `DAYS_ID_PUBLISH` approximates real-world deployment conditions.
- `class_weight='balanced'` was used to handle class imbalance, with a King-Zeng intercept correction applied post-training to restore calibrated probability outputs.

### Scorecard Scaling
- PDO = 20 | Target Score = 600 at odds 50:1 | Factor = 28.85 | Offset = 487.12

### Results
| Metric | Value |
| :--- | :--- |
| AUC | 0.7188 |
| KS | 0.3292 |
| Gini | 0.4376 |

---

## Model 2: XGBoost Challenger (`xgboost_challenger.py`)

### Experimental Design
Two XGBoost variants were trained to isolate algorithm contribution from information contribution:

- **Option A** - Same five features as the logistic model, raw (unprocessed) form. Controls for feature set; isolates algorithm effect.
- **Option B** - Same five features plus seven bureau-derived features aggregated from `bureau.csv`. Tests whether additional information improves performance beyond the algorithm gain.

This design answers a specific question: how much of any performance gain is due to the algorithm, and how much is due to having more information?

### Bureau Feature Engineering (Option B only)
Bureau records were filtered to `(DAYS_CREDIT <= 0) & (DAYS_CREDIT > -365)` before aggregation - excluding future-dated records (leakage prevention) and restricting to the last 12 months of history (recency window).

| Feature | Source Column | Aggregation | Missing Value Treatment |
| :--- | :--- | :--- | :--- |
| BUREAU_LOAN_COUNT | SK_ID_BUREAU | count | Fill 0 |
| BUREAU_AMT_CREDIT_SUM | AMT_CREDIT_SUM | sum | Fill 0 |
| BUREAU_AMT_OVERDUE_SUM | AMT_CREDIT_SUM_OVERDUE | sum | Fill 0 |
| BUREAU_AMT_OVERDUE_MAX | AMT_CREDIT_MAX_OVERDUE | max | Leave NaN |
| BUREAU_DAYS_OVERDUE_MAX | CREDIT_DAY_OVERDUE | max | Leave NaN |
| BUREAU_DELINQUENT_COUNT | CREDIT_DAY_OVERDUE >= 90 | count | Fill 0 |
| BUREAU_NO_HISTORY | Zero rows after filter | flag (0/1) | N/A |

Count and sum features fill to zero for no-history applicants because a sum over zero records is a known value. Max features remain NaN because a max over an empty set is genuinely undefined, and XGBoost's native NaN handling learns a default split direction for these applicants separately.

### Training Design
- Option A trained on the full development sample (246,010 rows).
- Option B trained on 85% of the development sample (209,108 rows), with 15% held out as an independent calibration set for `CalibratedClassifierCV(FrozenEstimator(...), method='sigmoid')`.
- Hyperparameter tuning: `RandomizedSearchCV` with `StratifiedKFold(n_splits=5)`, 50 iterations, scored on ROC AUC.
- `scale_pos_weight` computed separately per training population as `n_negative / n_positive`.

### Results

| Model | AUC | KS | Features | Data Sources |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.7188 | 0.3292 | 5 (WoE) | Application |
| XGBoost Option A | 0.7282 | 0.3395 | 5 (raw) | Application |
| XGBoost Option B | 0.7294 | 0.3435 | 12 (raw) | Application + Bureau |

Logistic regression metrics in this table reflect the canonical numbers from `scorecard_pipeline.py`. See Known Limitations for a note on the minor numerical discrepancy between scripts.

Calibrated mean predicted default probability: **9.15%** vs observed test default rate: **9.99%**.

### Key Findings

**Algorithm effect:** XGBoost-A outperforms the logistic model by 0.0094 AUC on identical features. This is a pure algorithm contribution with all other variables held constant.

**Information effect:** Adding seven bureau features increases AUC by only 0.0012 over XGBoost-A. The existing EXT_SOURCE variables likely already proxy bureau risk, leaving limited incremental signal for bureau features to contribute.

**Recommendation:** XGBoost-B is not recommended as a replacement for the primary scorecard. The 0.0106 total AUC gain over logistic regression does not justify the additional costs: full black-box status with no points table, a second data source requiring independent validation and refresh, and a calibration pipeline requiring ongoing maintenance. In an Indian NBFC context operating under RBI model risk guidelines, explainability requirements for a primary credit scorecard would not be met by an XGBoost model without significant additional model governance work. XGBoost-B may be suitable as a second-stage override model or internal risk-tiering tool where regulatory explainability requirements are lighter.

---

## Data Limitations
- 14.3% of applicants were thin-file customers with no prior bureau credit history.
- No true application timestamp existed; `DAYS_ID_PUBLISH` was used as a proxy for temporal ordering.
- Dataset class imbalance: approximately 92:8 non-default to default.
- EXT_SOURCE variables are undocumented by Home Credit. In production, feature provenance documentation would be required before regulatory approval.

---

## Known Limitations
- The proxy temporal split may not perfectly capture real-world temporal ordering.
- The logistic regression baseline inside `xgboost_challenger.py` is re-run after the bureau merge has been applied to `train_df`, which reorders rows and causes OptimalBinning to produce marginally different bin boundaries. This produces a ~0.0008 AUC difference (0.7188 vs 0.7180) between the two scripts for the same model. Canonical logistic metrics are those from `scorecard_pipeline.py`. This is a known reproducibility limitation of the current implementation.
- XGBoost Option A and Option B trained on different sample sizes due to the calibration holdout carved out exclusively for Option B, which may contribute marginally to the measured performance difference between the two XGBoost variants.
- A ~32% bad-rate prevalence gap exists between development (7.59%) and validation (9.99%) samples. Notably, the calibrated model's mean predicted probability (9.15%) tracks closer to the validation rate than the development rate, suggesting the model responds to population shift through feature values rather than being blind to it.
- Hyperparameter search reached the upper boundary for `min_child_weight`, indicating the search space may not have fully covered the optimal region.
- Feature attribution analysis (SHAP) was not conducted. The relative contribution of individual features - particularly whether EXT_SOURCE_2/3 dominate signal - remains unquantified and represents a recommended next step.

---

## How to Run

```bash
pip install -r requirements.txt

# Run the production logistic scorecard
python scorecard_pipeline.py

# Run the XGBoost challenger comparison
python xgboost_challenger.py
```

Outputs are written to `outputs/`: `scorecard_results.csv`, `scorecard_points_table.csv`.