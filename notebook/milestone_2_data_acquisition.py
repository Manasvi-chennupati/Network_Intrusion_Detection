# ============================================================
# MILESTONE 2 — DATA ACQUISITION
# XAI Feature Importance for Intrusion Detection
# ============================================================

import sys

# ------------------------------------------------------------
# 1. Python Environment Check
# ------------------------------------------------------------

print("=" * 60)
print("ENVIRONMENT CHECK")
print("=" * 60)

print("Python version:")
print(sys.version)


# ------------------------------------------------------------
# 2. Import Required Libraries
# ------------------------------------------------------------

from datasets import load_dataset
from sklearn.model_selection import train_test_split 

print("\nRequired libraries imported successfully!")


# ------------------------------------------------------------
# 3. Download NSL-KDD
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DOWNLOADING NSL-KDD")
print("=" * 60)

nsl_train = load_dataset(
    "Mireu-Lab/NSL-KDD",
    split="train"
).to_pandas()

nsl_test = load_dataset(
    "Mireu-Lab/NSL-KDD",
    split="test"
).to_pandas()

print("\nNSL-KDD downloaded successfully!")

print("NSL-KDD training shape:", nsl_train.shape)
print("NSL-KDD testing shape :", nsl_test.shape)
# Save NSL-KDD train/test datasets
nsl_train.to_csv(
    "data/nsl_kdd_train.csv",
    index=False
)

nsl_test.to_csv(
    "data/nsl_kdd_test.csv",
    index=False
)

print("\nNSL-KDD train/test files saved to data/")


# ------------------------------------------------------------
# 4. Inspect NSL-KDD
# ------------------------------------------------------------

print("\nNSL-KDD columns:")
print(nsl_train.columns.tolist())

print("\nNSL-KDD first 5 rows:")
print(nsl_train.head())


# ------------------------------------------------------------
# 5. Download RT-IoT2022
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DOWNLOADING RT-IoT2022")
print("=" * 60)

rt_iot_full = load_dataset(
    "michaelmallari/rt-iot2022",
    split="train"
).to_pandas()

print("\nRT-IoT2022 downloaded successfully!")

print("RT-IoT2022 shape:", rt_iot_full.shape)


# ------------------------------------------------------------
# 6. Inspect RT-IoT2022
# ------------------------------------------------------------

print("\nRT-IoT2022 columns:")
print(rt_iot_full.columns.tolist())

print("\nRT-IoT2022 first 5 rows:")
print(rt_iot_full.head())


# ------------------------------------------------------------
# 7. Find Possible Target Columns
# ------------------------------------------------------------

possible_targets = [
    column
    for column in rt_iot_full.columns
    if any(
        word in column.lower()
        for word in ["label", "class", "target"]
    )
]

print("\nPossible RT-IoT2022 target columns:")
print(possible_targets)


# ------------------------------------------------------------
# 8. Data Types
# ------------------------------------------------------------

print("\nRT-IoT2022 data types:")
print(rt_iot_full.dtypes)

# ------------------------------------------------------------
# 9. Create RT-IoT2022 80/20 Stratified Split
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CREATING RT-IoT2022 80/20 STRATIFIED SPLIT")
print("=" * 60)

rt_train, rt_test = train_test_split(
    rt_iot_full,
    test_size=0.20,
    stratify=rt_iot_full["Attack_type"],
    random_state=42
)

print("\nRT-IoT2022 split completed!")

print("RT-IoT2022 training shape:", rt_train.shape)
print("RT-IoT2022 testing shape :", rt_test.shape)


# ------------------------------------------------------------
# 10. Verify Stratification
# ------------------------------------------------------------

print("\nTraining class distribution:")
print(
    rt_train["Attack_type"]
    .value_counts(normalize=True)
    .sort_index()
)

print("\nTesting class distribution:")
print(
    rt_test["Attack_type"]
    .value_counts(normalize=True)
    .sort_index()
)


# ------------------------------------------------------------
# 11. Save Dataset Splits
# ------------------------------------------------------------

rt_train.to_csv(
    "data/rt_iot2022_train.csv",
    index=False
)

rt_test.to_csv(
    "data/rt_iot2022_test.csv",
    index=False
)

print("\nRT-IoT2022 train/test files saved to data/")


# ------------------------------------------------------------
# END OF MILESTONE 2
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MILESTONE 2 COMPLETED")
print("=" * 60)


# ------------------------------------------------------------
# END OF CURRENT STEP
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATA ACQUISITION AND INITIAL INSPECTION COMPLETED")
print("=" * 60)