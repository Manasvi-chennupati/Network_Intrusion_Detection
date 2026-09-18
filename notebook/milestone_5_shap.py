# ============================================================
# MILESTONE 5 — SHAP GLOBAL FEATURE IMPORTANCE
# ============================================================
# Requirement:
# Generate SHAP global importance via TreeSHAP over the
# FULL TEST SET and rank features by mean |SHAP value|.
#
# Datasets:
#   1. NSL-KDD
#   2. RT-IoT2022
#
# Important:
# RT-IoT2022 "Unnamed: 0" is explicitly excluded.
# ============================================================

import os
import pandas as pd
import numpy as np

from xgboost import XGBClassifier, DMatrix


# ============================================================
# SETTINGS
# ============================================================

os.makedirs("results", exist_ok=True)

xgb_params = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist"
}


# ============================================================
# FUNCTION:
# Calculate TreeSHAP global importance
# ============================================================

def calculate_tree_shap_importance(model, X_test):

    booster = model.get_booster()

    # XGBoost native DMatrix
    dmatrix = DMatrix(X_test)

    # Native TreeSHAP contributions
    contributions = booster.predict(
        dmatrix,
        pred_contribs=True
    )

    print("Raw SHAP contribution shape:", contributions.shape)

    # --------------------------------------------------------
    # Binary classification
    # Shape:
    # (samples, features + 1)
    # Last column = bias/base value
    # --------------------------------------------------------

    if contributions.ndim == 2:

        shap_values = contributions[:, :-1]

        mean_abs_shap = np.abs(
            shap_values
        ).mean(axis=0)

    # --------------------------------------------------------
    # Multiclass classification
    #
    # Shape:
    # (samples, classes, features + 1)
    #
    # Remove bias column and average absolute SHAP
    # over samples AND classes.
    # --------------------------------------------------------

    elif contributions.ndim == 3:

        shap_values = contributions[:, :, :-1]

        mean_abs_shap = np.abs(
            shap_values
        ).mean(axis=(0, 1))

    else:
        raise ValueError(
            f"Unexpected SHAP contribution shape: "
            f"{contributions.shape}"
        )

    # Safety check
    if len(mean_abs_shap) != len(X_test.columns):
        raise ValueError(
            "Number of SHAP importance values does not "
            "match number of features."
        )

    # Create ranking
    importance = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": mean_abs_shap
    })

    importance = importance.sort_values(
        by="mean_abs_shap",
        ascending=False
    ).reset_index(drop=True)

    importance["rank"] = (
        importance.index + 1
    )

    importance = importance[
        [
            "rank",
            "feature",
            "mean_abs_shap"
        ]
    ]

    return importance


# ============================================================
# PART 1 — NSL-KDD
# ============================================================

print("\n" + "=" * 70)
print("MILESTONE 5 — NSL-KDD TREE SHAP")
print("=" * 70)

# Load encoded data
nsl_X_train = pd.read_csv(
    "data/nsl_kdd_X_train_encoded.csv"
)

nsl_X_test = pd.read_csv(
    "data/nsl_kdd_X_test_encoded.csv"
)

nsl_y_train = pd.read_csv(
    "data/nsl_kdd_y_train_encoded.csv"
).squeeze()

nsl_y_test = pd.read_csv(
    "data/nsl_kdd_y_test_encoded.csv"
).squeeze()

print("Training shape:", nsl_X_train.shape)
print("Full test shape:", nsl_X_test.shape)

# Train fixed XGBoost model
nsl_model = XGBClassifier(
    **xgb_params,
    objective="binary:logistic"
)

nsl_model.fit(
    nsl_X_train,
    nsl_y_train
)

print("NSL-KDD XGBoost model trained.")

# Generate TreeSHAP importance
nsl_importance = calculate_tree_shap_importance(
    nsl_model,
    nsl_X_test
)

# Save complete ranking
nsl_output = (
    "results/"
    "nsl_kdd_shap_global_importance.csv"
)

nsl_importance.to_csv(
    nsl_output,
    index=False
)

print("\nNSL-KDD SHAP results saved to:")
print(nsl_output)

print("\nNSL-KDD TOP 10 FEATURES:")
print(
    nsl_importance.head(10).to_string(
        index=False
    )
)


# ============================================================
# PART 2 — RT-IoT2022
# ============================================================

print("\n" + "=" * 70)
print("MILESTONE 5 — RT-IoT2022 TREE SHAP")
print("=" * 70)

# Load encoded data
rt_X_train = pd.read_csv(
    "data/rt_iot2022_X_train_encoded.csv"
)

rt_X_test = pd.read_csv(
    "data/rt_iot2022_X_test_encoded.csv"
)

rt_y_train = pd.read_csv(
    "data/rt_iot2022_y_train_encoded.csv"
).squeeze()

rt_y_test = pd.read_csv(
    "data/rt_iot2022_y_test_encoded.csv"
).squeeze()

# ------------------------------------------------------------
# IMPORTANT LEAKAGE CHECK
# ------------------------------------------------------------

if "Unnamed: 0" in rt_X_train.columns:

    raise ValueError(
        "\nERROR: 'Unnamed: 0' is still present in "
        "RT-IoT2022 encoded training data.\n\n"
        "Regenerate Milestone 3 encoded files after "
        "dropping 'Unnamed: 0'."
    )

if "Unnamed: 0" in rt_X_test.columns:

    raise ValueError(
        "\nERROR: 'Unnamed: 0' is still present in "
        "RT-IoT2022 encoded test data.\n\n"
        "Regenerate Milestone 3 encoded files after "
        "dropping 'Unnamed: 0'."
    )

print("Leakage check: PASSED")
print("'Unnamed: 0' is NOT present.")

print("\nTraining shape:", rt_X_train.shape)
print("Full test shape:", rt_X_test.shape)

# Number of classes
rt_n_classes = rt_y_train.nunique()

print("Number of classes:", rt_n_classes)

# Train fixed multiclass XGBoost model
rt_model = XGBClassifier(
    **xgb_params,
    objective="multi:softprob",
    num_class=rt_n_classes
)

rt_model.fit(
    rt_X_train,
    rt_y_train
)

print("RT-IoT2022 XGBoost model trained.")

# Generate TreeSHAP importance
rt_importance = calculate_tree_shap_importance(
    rt_model,
    rt_X_test
)

# Save complete ranking
rt_output = (
    "results/"
    "rt_iot2022_shap_global_importance.csv"
)

rt_importance.to_csv(
    rt_output,
    index=False
)

print("\nRT-IoT2022 SHAP results saved to:")
print(rt_output)

print("\nRT-IoT2022 TOP 10 FEATURES:")
print(
    rt_importance.head(10).to_string(
        index=False
    )
)


# ============================================================
# CREATE VERIFICATION SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CREATING SHAP VERIFICATION SUMMARY")
print("=" * 70)

nsl_top10 = (
    nsl_importance
    .head(10)
    .copy()
)

nsl_top10.insert(
    0,
    "dataset",
    "NSL-KDD"
)

rt_top10 = (
    rt_importance
    .head(10)
    .copy()
)

rt_top10.insert(
    0,
    "dataset",
    "RT-IoT2022"
)

shap_top10_summary = pd.concat(
    [
        nsl_top10,
        rt_top10
    ],
    ignore_index=True
)

summary_output = (
    "results/"
    "milestone_5_shap_top10_summary.csv"
)

shap_top10_summary.to_csv(
    summary_output,
    index=False
)

print("\nVerification summary saved to:")
print(summary_output)


# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("MILESTONE 5 VERIFICATION")
print("=" * 70)

print("\nNSL-KDD:")
print(
    nsl_importance.head(10).to_string(
        index=False
    )
)

print("\nRT-IoT2022:")
print(
    rt_importance.head(10).to_string(
        index=False
    )
)

# Check feature counts
print("\nFeature counts:")
print(
    "NSL-KDD:",
    len(nsl_importance)
)

print(
    "RT-IoT2022:",
    len(rt_importance)
)

# Check duplicate features
print("\nDuplicate feature check:")

print(
    "NSL-KDD duplicates:",
    nsl_importance["feature"].duplicated().sum()
)

print(
    "RT-IoT2022 duplicates:",
    rt_importance["feature"].duplicated().sum()
)

print("\n" + "=" * 70)
print("MILESTONE 5 COMPLETED")
print("=" * 70)