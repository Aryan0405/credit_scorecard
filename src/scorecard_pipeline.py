import os
import numpy as np
import pandas as pd
from optbinning import OptimalBinning
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

def load_and_process_data(app_path):
    """Load application data and fix DAYS_EMPLOYED anomalies."""

    df = pd.read_csv(app_path)

    df['DAYS_EMPLOYED'] = (
        df['DAYS_EMPLOYED']
        .replace(365243, np.nan)
    )

    return df

def perform_time_split(df, date_col='DAYS_ID_PUBLISH', test_size=0.2):
    """Create temporal train/test split."""
    split_threshold = df[date_col].quantile(1 - test_size)
    train = df[df[date_col] <= split_threshold].copy()
    test = df[df[date_col] > split_threshold].copy()
    return train, test

def compute_woe_features(train, test, features):
    """Fit bins and transform variables to WoE."""
    binning_dict = {}

    for feature in features:
        optb = OptimalBinning(name=feature, dtype="numerical", solver="cp")
        optb.fit(train[feature], train["TARGET"])

        train[feature + "_WoE"] = optb.transform(train[feature], metric="woe")
        test[feature + "_WoE"] = optb.transform(test[feature], metric="woe")

        binning_dict[feature] = optb

    return train, test, binning_dict

def train_scorecard_model(X_train, y_train):
    """Train logistic regression scorecard."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def compute_scores(model, X_test, PDO=20, target_score=600, target_odds=50):
    """Convert probabilities to credit scores."""
    prediction_probabilities = model.predict_proba(X_test)[:, 1]

    Factor = PDO / np.log(2)
    offset = target_score - Factor * np.log(target_odds)

    scores = offset + Factor * np.log(
        (1 - prediction_probabilities) / prediction_probabilities
    )

    return scores

def build_scorecard_table(
    model,
    binning_dict,
    features,
    PDO=20,
    target_score=600,
    target_odds=50
):
    factor = PDO / np.log(2)
    offset = target_score - factor * np.log(target_odds)

    rows = []

    base_points = offset - factor * model.intercept_[0]

    rows.append({
        "Feature": "Base",
        "Bin": "N/A",
        "WoE": np.nan,
        "Coefficient": round(float(model.intercept_[0]), 4),
        "Points": round(base_points, 2)
    })

    for feature, coef in zip(features, model.coef_[0]):
        bin_table = binning_dict[feature].binning_table.build()

        for _, row in bin_table.iterrows():
            woe = pd.to_numeric(row.get("WoE"), errors="coerce")

            if pd.isna(woe):
                continue

            rows.append({
                "Feature": feature,
                "Bin": str(row["Bin"]),
                "WoE": round(woe, 4),
                "Coefficient": round(coef, 4),
                "Points": round(-coef * woe * factor, 2)
            })

    return pd.DataFrame(rows)

def calibrate_intercept(model, true_prevalence):
    """
    Correct intercept for class_weight='balanced' distortion.
    Restores predict_proba() outputs to reflect the true population
    default rate, not the artificial 50/50 the model was trained on.
    """
    effective_odds = 1.0  # 0.5 / (1 - 0.5) under 'balanced' weighting
    true_odds = true_prevalence / (1 - true_prevalence)

    correction = np.log(true_odds) - np.log(effective_odds)
    model.intercept_[0] += correction

    return model

if __name__ == "__main__":
    df = load_and_process_data('data/raw/application_train.csv')

    train, test = perform_time_split(df)

    features_to_woe = [
        'DAYS_BIRTH',
        'DAYS_EMPLOYED',
        'EXT_SOURCE_1',
        'EXT_SOURCE_2',
        'EXT_SOURCE_3'
    ]

    train, test, binning_dict = compute_woe_features(
        train,
        test,
        features_to_woe
    )

    X_train = train[[f + "_WoE" for f in features_to_woe]]
    y_train = train['TARGET']

    model = train_scorecard_model(X_train, y_train)

    model = calibrate_intercept(model, y_train.mean())

    X_test = test[[f + "_WoE" for f in features_to_woe]]

    prediction_probabilities = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(test['TARGET'], prediction_probabilities)
    print(f"ROC AUC Score: {auc:.4f}")

    ks_statistic, ks_p_value = ks_2samp(
        prediction_probabilities[test['TARGET'] == 0],
        prediction_probabilities[test['TARGET'] == 1]
    )

    print(
        f"KS Statistic: {ks_statistic:.4f}, "
        f"P-Value: {ks_p_value:.4f}"
    )

    gini_score = 2 * auc - 1
    print(f"Gini Coefficient: {gini_score:.4f}")

    os.makedirs("outputs", exist_ok=True)

    scores = compute_scores(model, X_test)
    test['SCORE'] = scores

    test[['SK_ID_CURR', 'TARGET', 'SCORE']].to_csv(
        'outputs/scorecard_results.csv',
        index=False
    )

    scorecard_table = build_scorecard_table(
        model,
        binning_dict,
        features_to_woe
    )

    scorecard_table.to_csv(
        'outputs/scorecard_points_table.csv',
        index=False
    )

    print("Saved outputs/scorecard_results.csv")
    print("Saved outputs/scorecard_points_table.csv")
    print(f"Train Prevalence: {y_train.mean():.4f}")
    print(f"Test Target Rate: {test['TARGET'].mean():.4f}")
    print(f"Mean Prediction Probability: {prediction_probabilities.mean():.4f}")