# ============================================================
# MILESTONE 4 — FIXED XGBOOST BASELINE
# XAI Feature Importance for Intrusion Detection
# ============================================================


# ============================================================
# STEP 1 — IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import f1_score


# ============================================================
# STEP 2 — LOAD ENCODED NSL-KDD DATA
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

nsl_y_test = pd.read_csv(
    "data/nsl_kdd_y_test_encoded.csv"
).squeeze()

print("NSL-KDD training features:", nsl_X_train.shape)
print("NSL-KDD testing features :", nsl_X_test.shape)
print("NSL-KDD training labels  :", nsl_y_train.shape)
print("NSL-KDD testing labels   :", nsl_y_test.shape)


# ============================================================
# STEP 3 — LOAD ENCODED RT-IoT2022 DATA
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

rt_y_test = pd.read_csv(
    "data/rt_iot2022_y_test_encoded.csv"
).squeeze()

print("RT-IoT2022 training features:", rt_X_train.shape)
print("RT-IoT2022 testing features :", rt_X_test.shape)
print("RT-IoT2022 training labels  :", rt_y_train.shape)
print("RT-IoT2022 testing labels   :", rt_y_test.shape)


# ============================================================
# STEP 4 — VERIFY DATA
# ============================================================

print("\n" + "=" * 60)
print("VERIFYING DATA")
print("=" * 60)

print(
    "NSL-KDD feature columns match:",
    list(nsl_X_train.columns) == list(nsl_X_test.columns)
)

print(
    "RT-IoT2022 feature columns match:",
    list(rt_X_train.columns) == list(rt_X_test.columns)
)

print(
    "NSL-KDD missing values:",
    nsl_X_train.isnull().sum().sum()
)

print(
    "RT-IoT2022 missing values:",
    rt_X_train.isnull().sum().sum()
)


# ============================================================
# STEP 5 — DEFINE FIXED XGBOOST HYPERPARAMETERS
# ============================================================
#
# The same hyperparameters are used for both datasets.
# Only the classification objective and number of classes
# differ according to the dataset.
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
# STEP 6 — TRAIN FIXED XGBOOST MODEL ON NSL-KDD
# ============================================================

print("\n" + "=" * 60)
print("TRAINING XGBOOST — NSL-KDD")
print("=" * 60)

nsl_n_classes = nsl_y_train.nunique()

print("Number of NSL-KDD classes:", nsl_n_classes)

nsl_model = XGBClassifier(
    **xgb_params,
    objective="binary:logistic"
)

nsl_model.fit(
    nsl_X_train,
    nsl_y_train
)

print("NSL-KDD XGBoost training completed.")


# ============================================================
# STEP 7 — EVALUATE NSL-KDD MODEL
# ============================================================

print("\n" + "=" * 60)
print("EVALUATING NSL-KDD")
print("=" * 60)

nsl_y_pred = nsl_model.predict(nsl_X_test)

nsl_macro_f1 = f1_score(
    nsl_y_test,
    nsl_y_pred,
    average="macro"
)

print(
    "NSL-KDD baseline test Macro-F1:",
    nsl_macro_f1
)


# ============================================================
# STEP 8 — TRAIN FIXED XGBOOST MODEL ON RT-IoT2022
# ============================================================

print("\n" + "=" * 60)
print("TRAINING XGBOOST — RT-IoT2022")
print("=" * 60)

rt_n_classes = rt_y_train.nunique()

print("Number of RT-IoT2022 classes:", rt_n_classes)

rt_model = XGBClassifier(
    **xgb_params,
    objective="multi:softprob",
    num_class=rt_n_classes
)

rt_model.fit(
    rt_X_train,
    rt_y_train
)

print("RT-IoT2022 XGBoost training completed.")


# ============================================================
# STEP 9 — EVALUATE RT-IoT2022 MODEL
# ============================================================

print("\n" + "=" * 60)
print("EVALUATING RT-IoT2022")
print("=" * 60)

rt_y_pred = rt_model.predict(rt_X_test)

rt_macro_f1 = f1_score(
    rt_y_test,
    rt_y_pred,
    average="macro"
)

print(
    "RT-IoT2022 baseline test Macro-F1:",
    rt_macro_f1
)


# ============================================================
# STEP 10 — DISPLAY FINAL BASELINE RESULTS
# ============================================================

print("\n" + "=" * 60)
print("MILESTONE 4 — BASELINE RESULTS")
print("=" * 60)

print(
    f"NSL-KDD     | Test Macro-F1 = {nsl_macro_f1:.4f}"
)

print(
    f"RT-IoT2022  | Test Macro-F1 = {rt_macro_f1:.4f}"
)


# ============================================================
# STEP 11 — SAVE BASELINE RESULTS
# ============================================================

baseline_results = pd.DataFrame({
    "dataset": [
        "NSL-KDD",
        "RT-IoT2022"
    ],
    "test_macro_f1": [
        nsl_macro_f1,
        rt_macro_f1
    ]
})

baseline_results.to_csv(
    "results/milestone_4_baseline_results.csv",
    index=False
)

print("\nBaseline results saved to:")
print("results/milestone_4_baseline_results.csv")


# ============================================================
# END OF MILESTONE 4
# ============================================================

print("\n" + "=" * 60)
print("MILESTONE 4 COMPLETED")
print("=" * 60)