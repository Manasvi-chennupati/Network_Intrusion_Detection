# ============================================================
# MILESTONE 3 — DATA ENCODING
# XAI Feature Importance for Intrusion Detection
# ============================================================

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


# ============================================================
# PART A — RT-IoT2022
# ============================================================

print("=" * 60)
print("MILESTONE 3 — RT-IoT2022 ENCODING")
print("=" * 60)


# ------------------------------------------------------------
# 1. Load RT-IoT2022 train and test data
# ------------------------------------------------------------

rt_train = pd.read_csv("data/rt_iot2022_train.csv").drop(columns=["Unnamed: 0"])
rt_test = pd.read_csv("data/rt_iot2022_test.csv").drop(columns=["Unnamed: 0"])

print("\nRT-IoT2022 train shape:", rt_train.shape)
print("RT-IoT2022 test shape :", rt_test.shape)


# ------------------------------------------------------------
# 2. Separate features and target
# ------------------------------------------------------------

X_rt_train = rt_train.drop(columns=["Attack_type"])
y_rt_train = rt_train["Attack_type"]

X_rt_test = rt_test.drop(columns=["Attack_type"])
y_rt_test = rt_test["Attack_type"]

print("\nTarget column: Attack_type")


# ------------------------------------------------------------
# 3. Define categorical columns
# ------------------------------------------------------------

rt_categorical_cols = [
    "proto",
    "service"
]

print("\nRT-IoT2022 categorical columns:")
print(rt_categorical_cols)


# ------------------------------------------------------------
# 4. Create OneHotEncoder
# ------------------------------------------------------------

rt_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)


# ------------------------------------------------------------
# 5. Fit encoder ONLY on training data
# ------------------------------------------------------------

rt_encoder.fit(
    X_rt_train[rt_categorical_cols]
)

print("\nRT-IoT2022 encoder fitted on training data.")


# ------------------------------------------------------------
# 6. Transform categorical training data
# ------------------------------------------------------------

rt_train_encoded = rt_encoder.transform(
    X_rt_train[rt_categorical_cols]
)

print(
    "Encoded RT-IoT2022 training categorical shape:",
    rt_train_encoded.shape
)


# ------------------------------------------------------------
# 7. Transform categorical testing data
# ------------------------------------------------------------

rt_test_encoded = rt_encoder.transform(
    X_rt_test[rt_categorical_cols]
)

print(
    "Encoded RT-IoT2022 testing categorical shape:",
    rt_test_encoded.shape
)


# ------------------------------------------------------------
# 8. Identify numerical columns
# ------------------------------------------------------------

rt_numeric_cols = [
    col
    for col in X_rt_train.columns
    if col not in rt_categorical_cols
]

print("\nNumber of numerical columns:", len(rt_numeric_cols))


# ------------------------------------------------------------
# 9. Extract numerical columns
# ------------------------------------------------------------

rt_train_numeric = X_rt_train[
    rt_numeric_cols
].copy()

rt_test_numeric = X_rt_test[
    rt_numeric_cols
].copy()


# ------------------------------------------------------------
# 10. Convert encoded categorical arrays into DataFrames
# ------------------------------------------------------------

rt_encoded_feature_names = (
    rt_encoder.get_feature_names_out(
        rt_categorical_cols
    )
)

rt_train_categorical = pd.DataFrame(
    rt_train_encoded,
    columns=rt_encoded_feature_names,
    index=X_rt_train.index
)

rt_test_categorical = pd.DataFrame(
    rt_test_encoded,
    columns=rt_encoded_feature_names,
    index=X_rt_test.index
)


# ------------------------------------------------------------
# 11. Combine numerical + encoded categorical features
# ------------------------------------------------------------

X_rt_train_encoded = pd.concat(
    [
        rt_train_numeric,
        rt_train_categorical
    ],
    axis=1
)

X_rt_test_encoded = pd.concat(
    [
        rt_test_numeric,
        rt_test_categorical
    ],
    axis=1
)

print(
    "\nFinal RT-IoT2022 training shape:",
    X_rt_train_encoded.shape
)

print(
    "Final RT-IoT2022 testing shape:",
    X_rt_test_encoded.shape
)


# ------------------------------------------------------------
# 12. Encode target variable
# ------------------------------------------------------------

rt_label_encoder = LabelEncoder()

y_rt_train_encoded = (
    rt_label_encoder.fit_transform(y_rt_train)
)

y_rt_test_encoded = (
    rt_label_encoder.transform(y_rt_test)
)

print(
    "\nNumber of RT-IoT2022 target classes:",
    len(rt_label_encoder.classes_)
)

print(
    "RT-IoT2022 target classes:"
)

print(
    rt_label_encoder.classes_
)


# ------------------------------------------------------------
# 13. Verify RT-IoT2022 encoding
# ------------------------------------------------------------

print("\nRT-IoT2022 verification:")

print(
    "Train and test feature columns match:",
    list(X_rt_train_encoded.columns)
    == list(X_rt_test_encoded.columns)
)

print(
    "Missing values in train:",
    X_rt_train_encoded.isnull().sum().sum()
)

print(
    "Missing values in test:",
    X_rt_test_encoded.isnull().sum().sum()
)


# ------------------------------------------------------------
# 14. Save encoded RT-IoT2022 data
# ------------------------------------------------------------

X_rt_train_encoded.to_csv(
    "data/rt_iot2022_X_train_encoded.csv",
    index=False
)

X_rt_test_encoded.to_csv(
    "data/rt_iot2022_X_test_encoded.csv",
    index=False
)

pd.Series(
    y_rt_train_encoded,
    name="Attack_type"
).to_csv(
    "data/rt_iot2022_y_train_encoded.csv",
    index=False
)

pd.Series(
    y_rt_test_encoded,
    name="Attack_type"
).to_csv(
    "data/rt_iot2022_y_test_encoded.csv",
    index=False
)

print(
    "\nRT-IoT2022 encoded files saved successfully."
)


# ============================================================
# PART B — NSL-KDD
# ============================================================

print("\n" + "=" * 60)
print("MILESTONE 3 — NSL-KDD ENCODING")
print("=" * 60)


# ------------------------------------------------------------
# 15. Load NSL-KDD train and test data
# ------------------------------------------------------------

nsl_train = pd.read_csv(
    "data/nsl_kdd_train.csv"
)

nsl_test = pd.read_csv(
    "data/nsl_kdd_test.csv"
)

print(
    "\nNSL-KDD train shape:",
    nsl_train.shape
)

print(
    "NSL-KDD test shape:",
    nsl_test.shape
)


# ------------------------------------------------------------
# 16. Separate features and target
# ------------------------------------------------------------

X_nsl_train = nsl_train.drop(
    columns=["class"]
)

y_nsl_train = nsl_train["class"]

X_nsl_test = nsl_test.drop(
    columns=["class"]
)

y_nsl_test = nsl_test["class"]

print("\nTarget column: class")


# ------------------------------------------------------------
# 17. Define categorical columns
# ------------------------------------------------------------

nsl_categorical_cols = [
    "protocol_type",
    "service",
    "flag"
]

print(
    "\nNSL-KDD categorical columns:"
)

print(
    nsl_categorical_cols
)


# ------------------------------------------------------------
# 18. Create OneHotEncoder
# ------------------------------------------------------------

nsl_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)


# ------------------------------------------------------------
# 19. Fit encoder ONLY on training data
# ------------------------------------------------------------

nsl_encoder.fit(
    X_nsl_train[nsl_categorical_cols]
)

print(
    "\nNSL-KDD encoder fitted on training data."
)


# ------------------------------------------------------------
# 20. Transform categorical training data
# ------------------------------------------------------------

nsl_train_encoded = nsl_encoder.transform(
    X_nsl_train[nsl_categorical_cols]
)

print(
    "Encoded NSL-KDD training categorical shape:",
    nsl_train_encoded.shape
)


# ------------------------------------------------------------
# 21. Transform categorical testing data
# ------------------------------------------------------------

nsl_test_encoded = nsl_encoder.transform(
    X_nsl_test[nsl_categorical_cols]
)

print(
    "Encoded NSL-KDD testing categorical shape:",
    nsl_test_encoded.shape
)


# ------------------------------------------------------------
# 22. Identify numerical columns
# ------------------------------------------------------------

nsl_numeric_cols = [
    col
    for col in X_nsl_train.columns
    if col not in nsl_categorical_cols
]

print(
    "\nNumber of numerical columns:",
    len(nsl_numeric_cols)
)


# ------------------------------------------------------------
# 23. Extract numerical columns
# ------------------------------------------------------------

nsl_train_numeric = X_nsl_train[
    nsl_numeric_cols
].copy()

nsl_test_numeric = X_nsl_test[
    nsl_numeric_cols
].copy()


# ------------------------------------------------------------
# 24. Convert encoded categorical arrays into DataFrames
# ------------------------------------------------------------

nsl_encoded_feature_names = (
    nsl_encoder.get_feature_names_out(
        nsl_categorical_cols
    )
)

nsl_train_categorical = pd.DataFrame(
    nsl_train_encoded,
    columns=nsl_encoded_feature_names,
    index=X_nsl_train.index
)

nsl_test_categorical = pd.DataFrame(
    nsl_test_encoded,
    columns=nsl_encoded_feature_names,
    index=X_nsl_test.index
)


# ------------------------------------------------------------
# 25. Combine numerical + encoded categorical features
# ------------------------------------------------------------

X_nsl_train_encoded = pd.concat(
    [
        nsl_train_numeric,
        nsl_train_categorical
    ],
    axis=1
)

X_nsl_test_encoded = pd.concat(
    [
        nsl_test_numeric,
        nsl_test_categorical
    ],
    axis=1
)

print(
    "\nFinal NSL-KDD training shape:",
    X_nsl_train_encoded.shape
)

print(
    "Final NSL-KDD testing shape:",
    X_nsl_test_encoded.shape
)


# ------------------------------------------------------------
# 26. Encode target variable
# ------------------------------------------------------------

nsl_label_encoder = LabelEncoder()

y_nsl_train_encoded = (
    nsl_label_encoder.fit_transform(
        y_nsl_train
    )
)

y_nsl_test_encoded = (
    nsl_label_encoder.transform(
        y_nsl_test
    )
)

print(
    "\nNumber of NSL-KDD target classes:",
    len(nsl_label_encoder.classes_)
)

print(
    "NSL-KDD target classes:"
)

print(
    nsl_label_encoder.classes_
)


# ------------------------------------------------------------
# 27. Verify NSL-KDD encoding
# ------------------------------------------------------------

print("\nNSL-KDD verification:")

print(
    "Train and test feature columns match:",
    list(X_nsl_train_encoded.columns)
    == list(X_nsl_test_encoded.columns)
)

print(
    "Missing values in train:",
    X_nsl_train_encoded.isnull().sum().sum()
)

print(
    "Missing values in test:",
    X_nsl_test_encoded.isnull().sum().sum()
)


# ------------------------------------------------------------
# 28. Save encoded NSL-KDD data
# ------------------------------------------------------------

X_nsl_train_encoded.to_csv(
    "data/nsl_kdd_X_train_encoded.csv",
    index=False
)

X_nsl_test_encoded.to_csv(
    "data/nsl_kdd_X_test_encoded.csv",
    index=False
)

pd.Series(
    y_nsl_train_encoded,
    name="class"
).to_csv(
    "data/nsl_kdd_y_train_encoded.csv",
    index=False
)

pd.Series(
    y_nsl_test_encoded,
    name="class"
).to_csv(
    "data/nsl_kdd_y_test_encoded.csv",
    index=False
)

print(
    "\nNSL-KDD encoded files saved successfully."
)


# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 60)
print("MILESTONE 3 COMPLETED")
print("=" * 60)

print("\nRT-IoT2022:")
print(
    "X_train:",
    X_rt_train_encoded.shape
)
print(
    "X_test :",
    X_rt_test_encoded.shape
)

print("\nNSL-KDD:")
print(
    "X_train:",
    X_nsl_train_encoded.shape
)
print(
    "X_test :",
    X_nsl_test_encoded.shape
)

print("\nAll encoded datasets saved successfully.")