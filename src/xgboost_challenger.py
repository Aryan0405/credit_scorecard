import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, StratifiedKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from scipy.stats import ks_2samp
from sklearn.frozen import FrozenEstimator
import joblib
import os
os.makedirs("models", exist_ok=True)
from scorecard_pipeline import (
    load_and_process_data, 
    perform_time_split,
    compute_woe_features,
    train_scorecard_model,
    calibrate_intercept
)

def compute_bureau_features(bureau_data):
    bureau_data = bureau_data[
        (bureau_data.DAYS_CREDIT <= 0) &
        (bureau_data.DAYS_CREDIT > -365)
    ]

    bureau_agg = bureau_data.groupby('SK_ID_CURR').agg(
        BUREAU_LOAN_COUNT=('SK_ID_BUREAU', 'count'),
        BUREAU_AMT_CREDIT_SUM=('AMT_CREDIT_SUM', 'sum'),
        BUREAU_AMT_OVERDUE_SUM=('AMT_CREDIT_SUM_OVERDUE', 'sum'),
        BUREAU_AMT_OVERDUE_MAX=('AMT_CREDIT_MAX_OVERDUE', 'max'),
        BUREAU_DAYS_OVERDUE_MAX=('CREDIT_DAY_OVERDUE', 'max'),
        BUREAU_DELINQUENT_COUNT=(
            'CREDIT_DAY_OVERDUE',
            lambda x: (x >= 90).sum()
        )
    ).reset_index()

    return bureau_agg

def add_bureau_features(application_data, bureau_data):
    bureau_agg = compute_bureau_features(bureau_data)

    application_data = application_data.merge(
        bureau_agg,
        on='SK_ID_CURR',
        how='left'
    )

    # NO HISTORY FLAG (based on merge result)
    application_data['BUREAU_NO_HISTORY'] = (
        application_data['BUREAU_LOAN_COUNT'].isna()
    ).astype(int)

    # Fill count and sum features
    application_data['BUREAU_LOAN_COUNT'] = application_data['BUREAU_LOAN_COUNT'].fillna(0)
    application_data['BUREAU_DELINQUENT_COUNT'] = application_data['BUREAU_DELINQUENT_COUNT'].fillna(0)
    application_data['BUREAU_AMT_CREDIT_SUM'] = application_data['BUREAU_AMT_CREDIT_SUM'].fillna(0)
    application_data['BUREAU_AMT_OVERDUE_SUM'] = application_data['BUREAU_AMT_OVERDUE_SUM'].fillna(0)

    # Leave magnitude features as NaN
    # (BUREAU_AMT_OVERDUE_MAX, BUREAU_DAYS_OVERDUE_MAX)

    return application_data

# Load application data using the existing pipeline function
application_data = load_and_process_data('data/raw/application_train.csv')

# Perform split immediately to secure original index/ordering
train_df_raw, test_df_raw = perform_time_split(application_data)

# 1. STRICT LOGISTIC REGRESSION BASELINE
# Executes on pristine data to mirror scorecard_pipeline.py exactly

features_to_woe = [
    'DAYS_BIRTH', 
    'DAYS_EMPLOYED', 
    'EXT_SOURCE_1', 
    'EXT_SOURCE_2', 
    'EXT_SOURCE_3'
]

# Compute WoE (Fits on pristine train, transforms both)
train_lr, test_lr, binning_dict = compute_woe_features(
    train_df_raw.copy(), 
    test_df_raw.copy(), 
    features_to_woe
)

# Isolate the WoE feature matrices and their specific targets
X_train_LR = train_lr[[f + "_WoE" for f in features_to_woe]]
y_train_LR = train_lr['TARGET']
X_test_LR = test_lr[[f + "_WoE" for f in features_to_woe]]
y_test_LR = test_lr['TARGET']
joblib.dump(y_test_LR,"models/y_test.pkl")
joblib.dump(X_train_LR,"models/X_train_LR.pkl")
joblib.dump(X_test_LR,"models/X_test_LR.pkl")

# Train and calibrate the Logistic Regression model
lr_model = train_scorecard_model(X_train_LR, y_train_LR)
lr_model = calibrate_intercept(lr_model, y_train_LR.mean())
joblib.dump(lr_model,"models/lr_model.pkl")

# Generate probabilities 
proba_logistic = lr_model.predict_proba(X_test_LR)[:, 1]

# Calculate metrics mapped to the exact indices of the pristine LR data
ks_statistic_LR, ks_p_value_LR = ks_2samp(
    proba_logistic[y_test_LR == 0], 
    proba_logistic[y_test_LR == 1]
)

print(f"Logistic Regression ROC AUC: {roc_auc_score(y_test_LR, proba_logistic):.4f}")

# 2. XGBOOST CHALLENGER MODELS
# Proceeds with bureau merges now that the LR baseline is secured

# Load bureau data independently
bureau_data = pd.read_csv('data/raw/bureau.csv')

# Modify dataframes for the XGBoost streams
train_df = add_bureau_features(train_df_raw.copy(), bureau_data)
test_df = add_bureau_features(test_df_raw.copy(), bureau_data)

# Define y_test for XGBoost (post-merge ordering)
y_test = test_df['TARGET']

# Carve out Calibration Holdout exclusively for Option B using a clean random split
train_tune_df, calib_df = train_test_split(
    train_df, 
    test_size=0.15, 
    stratify=train_df['TARGET'], 
    random_state=42
)

# Define independent targets
y_train_A = train_df['TARGET']          # Option A: Full training set
y_train_B = train_tune_df['TARGET']     # Option B: 85% tuning slice
y_calib_B = calib_df['TARGET']

# Compute distinct closed-form scale_pos_weights
scale_weight_A = (len(y_train_A) - y_train_A.sum()) / y_train_A.sum()
scale_weight_B = (len(y_train_B) - y_train_B.sum()) / y_train_B.sum()

print(f"Option A scale_pos_weight: {scale_weight_A:.4f}")
print(f"Option B scale_pos_weight: {scale_weight_B:.4f}")

# Build the strict feature matrices
features_A = [
    'DAYS_BIRTH', 
    'DAYS_EMPLOYED', 
    'EXT_SOURCE_1', 
    'EXT_SOURCE_2', 
    'EXT_SOURCE_3'
]
bureau_cols = ['BUREAU_LOAN_COUNT',
               'BUREAU_AMT_CREDIT_SUM',
               'BUREAU_AMT_OVERDUE_SUM',
               'BUREAU_AMT_OVERDUE_MAX',
               'BUREAU_DAYS_OVERDUE_MAX',
               'BUREAU_DELINQUENT_COUNT',
               'BUREAU_NO_HISTORY']

features_B = features_A + bureau_cols

# Option A matrices (Full Train)
X_train_A = train_df[features_A]
X_test_A = test_df[features_A]
joblib.dump(X_test_A,"models/X_test_A.pkl")

# Option B matrices (Tuned Train + Calib Holdout)
X_train_B = train_tune_df[features_B]
X_calib_B = calib_df[features_B]
X_test_B = test_df[features_B]
joblib.dump(X_test_B,"models/X_test_B.pkl")

# Base Estimator Setup (Independent weights)
base_xgb_A = XGBClassifier(
    scale_pos_weight=scale_weight_A,
    eval_metric='logloss',
    random_state=42,
    n_jobs=1
)

base_xgb_B = XGBClassifier(
    scale_pos_weight=scale_weight_B,
    eval_metric='logloss',
    random_state=42,
    n_jobs=1
)

param_grid = {
    'max_depth': [2, 3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [300, 500, 800],
    'min_child_weight': [50, 100, 200, 500, 800, 900, 1000],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_lambda': [1, 10, 50, 100, 200, 500, 1000]
}

# Explicit StratifiedKFold to guarantee shuffling and maintain class balance
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# RandomizedSearchCV Setup
search_A = RandomizedSearchCV(
    estimator=base_xgb_A,
    param_distributions=param_grid,
    n_iter=50,
    scoring='roc_auc',
    cv=cv_strategy, 
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search_B = RandomizedSearchCV(
    estimator=base_xgb_B,
    param_distributions=param_grid,
    n_iter=50,
    scoring='roc_auc',
    cv=cv_strategy, 
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search_A.fit(X_train_A, y_train_A)
joblib.dump(search_A.best_estimator_, "models/xgb_A.pkl")

search_B.fit(X_train_B, y_train_B)
joblib.dump(search_B.best_estimator_, "models/xgb_B.pkl")

print(f"Training set shape for Option A: {X_train_A.shape}")
print(f"Training set shape for Option B: {X_train_B.shape}")
print(f"Best parameters for Option A: {search_A.best_params_}")
print(f"Best score for Option A: {search_A.best_score_:.4f}")
print(f"Best parameters for Option B: {search_B.best_params_}")
print(f"Best score for Option B: {search_B.best_score_:.4f}")

proba_A = search_A.predict_proba(X_test_A)[:, 1]
proba_B = search_B.predict_proba(X_test_B)[:, 1]

print(f"Test A ROC AUC Score: {roc_auc_score(y_test, proba_A):.4f}")
print(f"Test B ROC AUC Score: {roc_auc_score(y_test, proba_B):.4f}")

calibrated_clf = CalibratedClassifierCV(FrozenEstimator(search_B.best_estimator_), method='sigmoid')
calibrated_clf.fit(X_calib_B, y_calib_B)

# Corrected mean calculation to only average the positive class probabilities
calibrated_probs = calibrated_clf.predict_proba(X_test_B)[:, 1].mean()

print(f"Calibrated probabilities mean: {calibrated_probs:.4f}")
print(f"Test set target proportion: {y_test.mean():.4f}")

ks_statistic_A, ks_p_value_A = ks_2samp(proba_A[y_test == 0], proba_A[y_test == 1])
ks_statistic_B, ks_p_value_B = ks_2samp(proba_B[y_test == 0], proba_B[y_test == 1])

print(f"KS Statistic for Option A: {ks_statistic_A:.4f}")
print(f"KS Statistic for Option B: {ks_statistic_B:.4f}")
print(f"KS Statistic for Logistic Regression: {ks_statistic_LR:.4f}")