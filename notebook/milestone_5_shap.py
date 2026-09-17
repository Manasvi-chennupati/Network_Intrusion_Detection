# ============================================================
# MILESTONE 5 — SHAP GLOBAL FEATURE IMPORTANCE
# XAI Feature Importance for Intrusion Detection
# ============================================================

import pandas as pd
import numpy as np
from xgboost import XGBClassifier, DMatrix


# ============================================================
# FIXED XGBOOST PARAMETERS — SAME AS MILESTONE 4
# ============================================================

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
# NSL-KDD
# ============================================================

print("=" * 60)
print("LOADING NSL-KDD DATA")
print("=" * 60)

nsl_X_train = pd.read_csv(
    "data/nsl_kdd_X_train_encoded.csv"
)

nsl_X_test = pd.read_csv(
    "data/nsl_kdd_X_test_encoded.csv"
)

nsl_y_train = pd.read_csv(
    "data/nsl_kdd_y_train_encoded.csv"
).squeeze()

print("NSL-KDD training shape:", nsl_X_train.shape)
print("NSL-KDD full test shape:", nsl_X_test.shape)


# ============================================================
# TRAIN NSL-KDD FIXED MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING FIXED NSL-KDD XGBOOST MODEL")
print("=" * 60)

nsl_model = XGBClassifier(
    **xgb_params,
    objective="binary:logistic"
)

nsl_model.fit(
    nsl_X_train,
    nsl_y_train
)

print("NSL-KDD model training completed.")


# ============================================================
# NSL-KDD TREE SHAP
# ============================================================

print("\n" + "=" * 60)
print("CALCULATING NSL-KDD TREE SHAP VALUES")
print("=" * 60)

nsl_booster = nsl_model.get_booster()

# XGBoost Booster.predict requires DMatrix
nsl_dmatrix = DMatrix(nsl_X_test)

# Native XGBoost TreeSHAP
nsl_contrib = nsl_booster.predict(
    nsl_dmatrix,
    pred_contribs=True
)

print("NSL-KDD contribution matrix shape:",
      nsl_contrib.shape)

# Last column is the bias/base-value contribution.
# Remove it because we only rank actual features.
nsl_shap_values = nsl_contrib[:, :-1]

print("NSL-KDD SHAP matrix shape:",
      nsl_shap_values.shape)


# ============================================================
# NSL-KDD GLOBAL IMPORTANCE
# mean(|SHAP|)
# ============================================================

nsl_shap_importance = pd.DataFrame({
    "feature": nsl_X_test.columns,
    "mean_abs_shap": np.abs(
        nsl_shap_values
    ).mean(axis=0)
})

nsl_shap_importance = (
    nsl_shap_importance
    .sort_values(
        by="mean_abs_shap",
        ascending=False
    )
    .reset_index(drop=True)
)

nsl_shap_importance["rank"] = (
    nsl_shap_importance.index + 1
)

nsl_shap_importance = nsl_shap_importance[
    [
        "rank",
        "feature",
        "mean_abs_shap"
    ]
]


# ============================================================
# SAVE NSL-KDD RESULTS
# ============================================================

nsl_shap_importance.to_csv(
    "results/nsl_kdd_shap_global_importance.csv",
    index=False
)

print("\nTop 10 NSL-KDD SHAP features:")
print(
    nsl_shap_importance
    .head(10)
    .to_string(index=False)
)

print(
    "\nSaved:"
    " results/nsl_kdd_shap_global_importance.csv"
)


# ============================================================
# RT-IoT2022
# ============================================================

print("\n" + "=" * 60)
print("LOADING RT-IoT2022 DATA")
print("=" * 60)

rt_X_train = pd.read_csv(
    "data/rt_iot2022_X_train_encoded.csv"
)

rt_X_test = pd.read_csv(
    "data/rt_iot2022_X_test_encoded.csv"
)

rt_y_train = pd.read_csv(
    "data/rt_iot2022_y_train_encoded.csv"
).squeeze()

print("RT-IoT2022 training shape:",
      rt_X_train.shape)

print("RT-IoT2022 full test shape:",
      rt_X_test.shape)


# ============================================================
# TRAIN RT-IoT2022 FIXED MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING FIXED RT-IoT2022 XGBOOST MODEL")
print("=" * 60)

rt_n_classes = rt_y_train.nunique()

rt_model = XGBClassifier(
    **xgb_params,
    objective="multi:softprob",
    num_class=rt_n_classes
)

rt_model.fit(
    rt_X_train,
    rt_y_train
)

print("RT-IoT2022 model training completed.")
print("Number of classes:", rt_n_classes)


# ============================================================
# RT-IoT2022 TREE SHAP
# ============================================================

print("\n" + "=" * 60)
print("CALCULATING RT-IoT2022 TREE SHAP VALUES")
print("=" * 60)

rt_booster = rt_model.get_booster()

# XGBoost Booster.predict requires DMatrix
rt_dmatrix = DMatrix(rt_X_test)

# Native XGBoost TreeSHAP
rt_contrib = rt_booster.predict(
    rt_dmatrix,
    pred_contribs=True
)

print(
    "Raw RT-IoT2022 contribution shape:",
    rt_contrib.shape
)


# ============================================================
# HANDLE RT-IoT2022 MULTICLASS SHAP OUTPUT
# ============================================================

if rt_contrib.ndim == 3:

    # Shape:
    # samples × classes × (features + bias)

    # Remove bias/base-value column
    rt_shap_values = rt_contrib[:, :, :-1]

    print(
        "RT-IoT2022 SHAP values shape:",
        rt_shap_values.shape
    )

    # Mean absolute SHAP:
    # 1. absolute value
    # 2. mean over samples
    # 3. mean over classes

    rt_shap_importance_values = (
        np.abs(rt_shap_values)
        .mean(axis=(0, 1))
    )

else:

    # Fallback if XGBoost returns 2D output

    # Remove bias/base-value column
    rt_shap_values = rt_contrib[:, :-1]

    print(
        "RT-IoT2022 SHAP values shape:",
        rt_shap_values.shape
    )

    rt_shap_importance_values = (
        np.abs(rt_shap_values)
        .mean(axis=0)
    )


# ============================================================
# RT-IoT2022 GLOBAL IMPORTANCE
# ============================================================

print(
    "RT-IoT2022 feature importance shape:",
    rt_shap_importance_values.shape
)

rt_shap_importance = pd.DataFrame({
    "feature": rt_X_test.columns,
    "mean_abs_shap": rt_shap_importance_values
})

rt_shap_importance = (
    rt_shap_importance
    .sort_values(
        by="mean_abs_shap",
        ascending=False
    )
    .reset_index(drop=True)
)

rt_shap_importance["rank"] = (
    rt_shap_importance.index + 1
)

rt_shap_importance = rt_shap_importance[
    [
        "rank",
        "feature",
        "mean_abs_shap"
    ]
]


# ============================================================
# SAVE RT-IoT2022 RESULTS
# ============================================================

rt_shap_importance.to_csv(
    "results/rt_iot2022_shap_global_importance.csv",
    index=False
)

print("\nTop 10 RT-IoT2022 SHAP features:")
print(
    rt_shap_importance
    .head(10)
    .to_string(index=False)
)

print(
    "\nSaved:"
    " results/rt_iot2022_shap_global_importance.csv"
)


# ============================================================
# MILESTONE 5 COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("MILESTONE 5 COMPLETED")
print("=" * 60)

print(
    "NSL-KDD:"
    " results/nsl_kdd_shap_global_importance.csv"
)

print(
    "RT-IoT2022:"
    " results/rt_iot2022_shap_global_importance.csv"
)