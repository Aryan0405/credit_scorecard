# Credit Risk Prediction Model: Home Credit Default Analysis

Credit Risk is the risk that a borrower fails to repay what they owe. Banks, NBFCs, and fintech lenders quantify this risk for two major reasons: (1) Pricing — charging higher interest rates to riskier borrowers, and (2) Capital Adequacy — regulators under the Basel Committee on Banking Supervision framework require banks to hold capital buffers against potential losses.

The objective of this project was to predict the probability of applicants experiencing payment difficulties according to Home Credit internal criteria (TARGET = 1).

## Features Used
Rigorous feature selection was conducted using Information Value (IV) to prioritize predictive power while maintaining business interpretability.

| Feature Name | Information Value (IV) |
| :--- | :--- |
| EXT_SOURCE_3 | 0.3282 |
| EXT_SOURCE_2 | 0.3119 |
| EXT_SOURCE_1 | 0.1396 |
| DAYS_EMPLOYED | 0.1093 |
| DAYS_BIRTH | 0.0853 |

## Data Limitations
* 14.3% of applicants were “thin-file” customers with no prior bureau credit history.
* The dataset did not contain a true application timestamp, so a proxy temporal split was created using DAYS_ID_PUBLISH.
* The dataset was highly imbalanced (92:8), meaning only 8% of applicants were defaulters.

## Methodology
* Used Weight of Evidence (WoE) binning to transform variables into monotonic risk relationships suitable for logistic regression scorecards.
* Feature selection was performed using Information Value (IV) and business interpretability to avoid unstable predictors and multicollinearity.
* A proxy time-based split using DAYS_ID_PUBLISH was used instead of a random split to better approximate real-world model deployment conditions.

## Scorecard Scaling
* PDO = 20
* Target Score = 600 at odds 50:1
* Factor = 28.85
* Offset = 487.12

## Results
* AUC = 0.7188
* KS = 0.3292
* Gini = 0.4376

The model demonstrated good discriminatory power, with a 71.9% probability of correctly ranking a defaulter above a non-defaulter (AUC), decent separation between risky and safe borrowers (KS), and overall strong scorecard performance (Gini).

## Known Limitations
* No true timestamp existed in the dataset, so the proxy temporal split using DAYS_ID_PUBLISH may not perfectly capture real-world temporal ordering.
* EXT_SOURCE variables had undocumented feature construction, creating potential leakage risk.
* EXT_SOURCE features were the strongest predictors(EXT_SOURCE_3 IV: 0.3282, EXT_SOURCE_2 IV: 0.3119) but their construction is undocumented by Home Credit. In a real deployment, using unexplainable external scores raises regulatory concerns under model governance frameworks. A production model would require feature provenance documentation before approval.
* The model intentionally used only five features for interpretability, which may have introduced underfitting.
* Bureau data was available for only 85.7% of applicants, while thin-file applicants were handled through imputation strategies.

## How to Run

```bash
pip install -r requirements.txt
python src/scorecard_pipeline.py