import numpy as np
import pandas as pd
from optbinning import OptimalBinning
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
def load_and_process_data(app_path, bureau_path):
    application_data = pd.read_csv(app_path)
    bureau_data = pd.read_csv(bureau_path)
    bureau_agg = bureau_data.groupby('SK_ID_CURR').agg(
        BUREAU_LOAN_COUNT=('SK_ID_BUREAU', 'count'),
        BUREAU_AMT_CREDIT_SUM=('AMT_CREDIT_SUM', 'sum'),
        BUREAU_AMT_OVERDUE_SUM=('AMT_CREDIT_SUM_OVERDUE', 'sum'),
        BUREAU_AMT_OVERDUE_MAX=('AMT_CREDIT_SUM_OVERDUE', 'max'),
        BUREAU_DAYS_OVERDUE_MAX=('CREDIT_DAY_OVERDUE', 'max'),
        BUREAU_DELINQUENT_COUNT=('CREDIT_DAY_OVERDUE', lambda x: (x > 0).sum())
    ).reset_index()
    df = application_data.merge(bureau_agg, on='SK_ID_CURR', how='left')
    df['BUREAU_NO_HISTORY'] = df['BUREAU_LOAN_COUNT'].isnull().astype(int)
    bureau_cols = ['BUREAU_LOAN_COUNT', 'BUREAU_AMT_CREDIT_SUM', 
                'BUREAU_AMT_OVERDUE_SUM', 'BUREAU_AMT_OVERDUE_MAX',
                'BUREAU_DAYS_OVERDUE_MAX', 'BUREAU_DELINQUENT_COUNT']

    df[bureau_cols] = df[bureau_cols].fillna(0)
    df['DAYS_EMPLOYED_ANOMALY'] = df['DAYS_EMPLOYED'].apply(lambda x: 1 if x == 365243 else 0)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    return df

def perform_time_split(df, date_col='DAYS_ID_PUBLISH', test_size=0.2):
    split_threshold = df[date_col].quantile(1 - test_size)
    train = df[df[date_col] <= split_threshold].copy()
    test = df[df[date_col] > split_threshold].copy()
    return train, test

def compute_woe_features(train,test,features):
    binning_dict = {}

    for feature in features:
        optb = OptimalBinning(name=feature, dtype="numerical", solver="cp")
        optb.fit(train[feature], train["TARGET"])

        train[feature + "_WoE"] = optb.transform(train[feature], metric="woe")
        test[feature + "_WoE"] = optb.transform(test[feature], metric="woe")

        binning_dict[feature] = optb
    return train,test,binning_dict

def train_scorecard_model(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000,class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def compute_scores(model,X_test,PDO=20,target_score=600,target_odds=50):
    prediction_probabilities = model.predict_proba(X_test)[:, 1]
    Factor = PDO / np.log(2)
    offset = target_score - Factor * np.log(target_odds)
    scores = offset + Factor * np.log((1 - prediction_probabilities) / prediction_probabilities)
    return scores

if __name__ == "__main__":
    df = load_and_process_data('application_data.csv', 'bureau_data.csv')
    train, test = perform_time_split(df)
    features_to_woe = ['DAYS_BIRTH','DAYS_EMPLOYED','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']
    train, test, binning_dict = compute_woe_features(train, test, features_to_woe)
    X_train = train[[f + "_WoE" for f in features_to_woe]]
    y_train = train['TARGET']
    model = train_scorecard_model(X_train, y_train)
    X_test = test[[f + "_WoE" for f in features_to_woe]]
    prediction_probabilities = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(test['TARGET'], prediction_probabilities)
    print(f"ROC AUC Score: {auc:.4f}")
    ks_statistic, ks_p_value = ks_2samp(prediction_probabilities[test['TARGET'] == 0], prediction_probabilities[test['TARGET'] == 1])
    print(f"KS Statistic: {ks_statistic:.4f}, P-Value: {ks_p_value:.4f}")
    gini_score = 2 * auc - 1
    print(f"Gini Coefficient: {gini_score:.4f}")
    scores = compute_scores(model, X_test)
    test['SCORE'] = scores