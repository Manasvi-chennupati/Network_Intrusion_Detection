import os
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from xgboost import XGBClassifier


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

REMOVAL_LEVELS = [0, 5, 10, 20, 30, 40, 50]

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
# Helper functions
# ============================================================

def load_importance(path):
    df = pd.read_csv(path)

    # Expected format:
    # feature + importance/frequency column
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


def evaluate_deletion(
    model,
    X_test,
    y_test,
    ranking,
    baseline_f1,
    dataset_name,
    method
):
    results = []

    # Copy test data so original data is never modified
    X_masked = X_test.copy()

    # Baseline: no features removed
    if 0 in REMOVAL_LEVELS:
        results.append({
            "dataset": dataset_name,
            "method": method,
            "removed_features": 0,
            "test_macro_f1": baseline_f1,
            "f1_drop": 0.0
        })

    for k in REMOVAL_LEVELS:
        if k == 0:
            continue

        selected_features = ranking[:k]

        # Validate feature names
        missing_features = [
            feature
            for feature in selected_features
            if feature not in X_masked.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing features for {dataset_name} / {method}: "
                f"{missing_features}"
            )

        # Mask selected features with zero
        X_masked = X_test.copy()
        X_masked.loc[:, selected_features] = 0

        predictions = model.predict(X_masked)

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro"
        )

        results.append({
            "dataset": dataset_name,
            "method": method,
            "removed_features": k,
            "test_macro_f1": macro_f1,
            "f1_drop": baseline_f1 - macro_f1
        })

    return results


# ============================================================
# Main
# ============================================================

all_results = []

os.makedirs("results", exist_ok=True)


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

    # Safety check for RT-IoT2022 leakage artifact
    if dataset_name == "RT-IoT2022":
        if "Unnamed: 0" in X_train.columns:
            raise ValueError(
                "RT-IoT2022 still contains 'Unnamed: 0'. "
                "Regenerate Milestone 3 encoded files."
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
    # Baseline test Macro-F1
    # --------------------------------------------------------

    baseline_predictions = model.predict(X_test)

    baseline_f1 = f1_score(
        y_test,
        baseline_predictions,
        average="macro"
    )

    print(f"Baseline Test Macro-F1: {baseline_f1:.6f}")

    # --------------------------------------------------------
    # Load SHAP / LIME / GWO rankings
    # --------------------------------------------------------

    rankings = {
        "SHAP": load_importance(paths["shap"]),
        "LIME": load_importance(paths["lime"]),
        "GWO": load_importance(paths["gwo"]),
    }

    # --------------------------------------------------------
    # Evaluate each explanation method
    # --------------------------------------------------------

    for method, ranking in rankings.items():

        print(f"\n{method} deletion evaluation")

        method_results = evaluate_deletion(
            model=model,
            X_test=X_test,
            y_test=y_test,
            ranking=ranking,
            baseline_f1=baseline_f1,
            dataset_name=dataset_name,
            method=method
        )

        all_results.extend(method_results)

        for row in method_results:
            print(
                f"Removed: {row['removed_features']:>2} | "
                f"Macro-F1: {row['test_macro_f1']:.6f} | "
                f"F1 drop: {row['f1_drop']:.6f}"
            )


# ============================================================
# Save results
# ============================================================

results_df = pd.DataFrame(all_results)

output_path = "results/milestone_8_deletion_faithfulness.csv"

results_df.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 70)
print("Milestone 8 completed.")
print(f"Results saved to: {output_path}")
print("=" * 70)

print("\nFinal results:")
print(results_df.to_string(index=False))