import os
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from xgboost import XGBClassifier


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42
THRESHOLD = 0.90

DATASETS = {
    "NSL-KDD": {
        "X_train": "data/nsl_kdd_X_train_encoded.csv",
        "y_train": "data/nsl_kdd_y_train_encoded.csv",
        "X_test": "data/nsl_kdd_X_test_encoded.csv",
        "y_test": "data/nsl_kdd_y_test_encoded.csv",
        "shap": "results/nsl_kdd_shap_global_importance.csv",
        "lime": "results/nsl_kdd_lime_global_importance.csv",
        "gwo": "results/nsl_kdd_gwo_global_importance.csv",
    },
    "RT-IoT2022": {
        "X_train": "data/rt_iot2022_X_train_encoded.csv",
        "y_train": "data/rt_iot2022_y_train_encoded.csv",
        "X_test": "data/rt_iot2022_X_test_encoded.csv",
        "y_test": "data/rt_iot2022_y_test_encoded.csv",
        "shap": "results/rt_iot2022_shap_global_importance.csv",
        "lime": "results/rt_iot2022_lime_global_importance.csv",
        "gwo": "results/rt_iot2022_gwo_global_importance.csv",
    },
}


XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "tree_method": "hist",
}


# ============================================================
# Functions
# ============================================================

def load_ranking(path):
    df = pd.read_csv(path)

    if "feature" not in df.columns:
        raise ValueError(f"'feature' column not found in {path}")

    return df["feature"].tolist()


def train_model(dataset_name, X_train, y_train):

    if dataset_name == "NSL-KDD":
        model = XGBClassifier(
            **XGB_PARAMS,
            objective="binary:logistic"
        )
    else:
        n_classes = len(np.unique(y_train))

        model = XGBClassifier(
            **XGB_PARAMS,
            objective="multi:softprob",
            num_class=n_classes
        )

    model.fit(X_train, y_train)

    return model


def calculate_sparsity(
    model,
    X_test,
    y_test,
    ranking,
    dataset_name,
    method,
    baseline_f1
):

    n_features = X_test.shape[1]

    # Start with every feature masked
    X_masked = pd.DataFrame(
        0,
        index=X_test.index,
        columns=X_test.columns
    )

    threshold_value = THRESHOLD * baseline_f1

    results = []

    # k = 0: completely masked
    predictions = model.predict(X_masked)

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    results.append({
        "dataset": dataset_name,
        "method": method,
        "k": 0,
        "macro_f1": macro_f1,
        "baseline_macro_f1": baseline_f1,
        "threshold_90_percent": threshold_value
    })

    # Add ranked features one by one
    for k in range(1, n_features + 1):

        feature = ranking[k - 1]

        if feature not in X_test.columns:
            raise ValueError(
                f"Feature '{feature}' from {method} ranking "
                f"not found in {dataset_name} test data."
            )

        # Reveal this feature
        X_masked.loc[:, feature] = X_test[feature]

        predictions = model.predict(X_masked)

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro"
        )

        results.append({
            "dataset": dataset_name,
            "method": method,
            "k": k,
            "macro_f1": macro_f1,
            "baseline_macro_f1": baseline_f1,
            "threshold_90_percent": threshold_value
        })

        # Stop at first k reaching 90%
        if macro_f1 >= threshold_value:

            print(
                f"{method}: 90% baseline reached at k={k} "
                f"(Macro-F1={macro_f1:.6f})"
            )

            return results, k, macro_f1

    print(
        f"{method}: 90% baseline NOT reached "
        f"with all {n_features} features."
    )

    return results, None, None


# ============================================================
# Main
# ============================================================

os.makedirs("results", exist_ok=True)

all_curve_results = []
summary_results = []


for dataset_name, paths in DATASETS.items():

    print("\n" + "=" * 70)
    print(f"Dataset: {dataset_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X_train = pd.read_csv(paths["X_train"])
    y_train = pd.read_csv(paths["y_train"]).squeeze()

    X_test = pd.read_csv(paths["X_test"])
    y_test = pd.read_csv(paths["y_test"]).squeeze()

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape : {X_test.shape}")

    # Leakage safety check
    if dataset_name == "RT-IoT2022":
        if "Unnamed: 0" in X_train.columns:
            raise ValueError(
                "RT-IoT2022 contains 'Unnamed: 0'. "
                "Regenerate clean Milestone 3 encoded data."
            )

    # --------------------------------------------------------
    # Train fixed XGBoost model
    # --------------------------------------------------------

    print("\nTraining fixed XGBoost model...")

    model = train_model(
        dataset_name,
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Full-feature baseline
    # --------------------------------------------------------

    baseline_predictions = model.predict(X_test)

    baseline_f1 = f1_score(
        y_test,
        baseline_predictions,
        average="macro"
    )

    print(
        f"Baseline Macro-F1: {baseline_f1:.6f}"
    )

    # --------------------------------------------------------
    # Load rankings
    # --------------------------------------------------------

    rankings = {
        "SHAP": load_ranking(paths["shap"]),
        "LIME": load_ranking(paths["lime"]),
        "GWO": load_ranking(paths["gwo"]),
    }

    # Make sure rankings contain every feature
    for method, ranking in rankings.items():

        ranking_set = set(ranking)
        feature_set = set(X_test.columns)

        missing = feature_set - ranking_set

        if missing:
            raise ValueError(
                f"{method} ranking is missing "
                f"{len(missing)} test features."
            )

    # --------------------------------------------------------
    # Calculate sparsity
    # --------------------------------------------------------

    for method, ranking in rankings.items():

        print(f"\n{method} — Addition/Sufficiency")

        curve, sparsity_k, f1_at_k = calculate_sparsity(
            model=model,
            X_test=X_test,
            y_test=y_test,
            ranking=ranking,
            dataset_name=dataset_name,
            method=method,
            baseline_f1=baseline_f1
        )

        all_curve_results.extend(curve)

        summary_results.append({
            "dataset": dataset_name,
            "method": method,
            "baseline_macro_f1": baseline_f1,
            "threshold_90_percent": THRESHOLD * baseline_f1,
            "sparsity_k": sparsity_k,
            "macro_f1_at_sparsity_k": f1_at_k
        })


# ============================================================
# Save outputs
# ============================================================

curve_df = pd.DataFrame(all_curve_results)

summary_df = pd.DataFrame(summary_results)

curve_df.to_csv(
    "results/milestone_8_sparsity_curve.csv",
    index=False
)

summary_df.to_csv(
    "results/milestone_8_sparsity.csv",
    index=False
)

print("\n" + "=" * 70)
print("Milestone 8 — Sparsity completed")
print("=" * 70)

print("\nSparsity summary:")
print(summary_df.to_string(index=False))

print("\nFiles saved:")
print("results/milestone_8_sparsity_curve.csv")
print("results/milestone_8_sparsity.csv")