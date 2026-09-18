import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from lime.lime_tabular import LimeTabularExplainer


# ============================================================
# MILESTONE 6 — LIME GLOBAL FEATURE IMPORTANCE
# ============================================================

RANDOM_STATE = 42
N_SAMPLES = 300

xgb_params = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "tree_method": "hist"
}


# ============================================================
# FUNCTION: GENERATE LIME GLOBAL IMPORTANCE
# ============================================================

def generate_lime_importance(
    X_train,
    X_test,
    y_train,
    model,
    output_file,
    dataset_name
):
    print("\n" + "=" * 60)
    print(f"MILESTONE 6 — {dataset_name} LIME")
    print("=" * 60)

    feature_names = list(X_train.columns)

    # --------------------------------------------------------
    # Create a fixed random sample of 300 test instances
    # --------------------------------------------------------

    rng = np.random.RandomState(RANDOM_STATE)

    sample_size = min(N_SAMPLES, len(X_test))

    sample_indices = rng.choice(
        len(X_test),
        size=sample_size,
        replace=False
    )

    X_sample = X_test.iloc[sample_indices].copy()

    print(f"Total test instances : {len(X_test)}")
    print(f"LIME sample size     : {sample_size}")

    # --------------------------------------------------------
    # LIME prediction function
    # --------------------------------------------------------

    def predict_fn(data):
        data_df = pd.DataFrame(
            data,
            columns=feature_names
        )
        return model.predict_proba(data_df)

    # --------------------------------------------------------
    # Create LimeTabularExplainer
    # --------------------------------------------------------

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=[
            str(class_name)
            for class_name in sorted(y_train.unique())
        ],
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # Store absolute local feature weights
    # --------------------------------------------------------

    feature_weight_sum = np.zeros(len(feature_names))

    # --------------------------------------------------------
    # Explain each of the 300 test instances
    # --------------------------------------------------------

    for i, (_, instance) in enumerate(X_sample.iterrows(), start=1):

        explanation = explainer.explain_instance(
            instance.values,
            predict_fn,
            num_features=len(feature_names)
        )

        # Convert LIME feature indices to weights
        local_weights = np.zeros(len(feature_names))

        for feature_index, weight in explanation.as_map()[
            explanation.available_labels()[0]
        ]:
            local_weights[feature_index] = weight

        # Accumulate absolute weights
        feature_weight_sum += np.abs(local_weights)

        if i % 25 == 0 or i == sample_size:
            print(f"Explained {i}/{sample_size} instances")

    # --------------------------------------------------------
    # Mean absolute LIME weight
    # --------------------------------------------------------

    mean_abs_lime = feature_weight_sum / sample_size

    lime_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_lime": mean_abs_lime
    })

    # Rank features
    lime_importance = (
        lime_importance
        .sort_values(
            by="mean_abs_lime",
            ascending=False
        )
        .reset_index(drop=True)
    )

    lime_importance["rank"] = (
        lime_importance.index + 1
    )

    lime_importance = lime_importance[
        ["rank", "feature", "mean_abs_lime"]
    ]

    # --------------------------------------------------------
    # Save complete ranking
    # --------------------------------------------------------

    lime_importance.to_csv(
        output_file,
        index=False
    )

    print("\nTop 10 LIME features:")
    print(lime_importance.head(10).to_string(index=False))

    print(f"\nSaved: {output_file}")

    return lime_importance


# ============================================================
# 1. NSL-KDD
# ============================================================

print("\nLoading NSL-KDD...")

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


# Train the same fixed XGBoost model
nsl_model = XGBClassifier(
    **xgb_params,
    objective="binary:logistic"
)

nsl_model.fit(
    nsl_X_train,
    nsl_y_train
)

nsl_lime = generate_lime_importance(
    X_train=nsl_X_train,
    X_test=nsl_X_test,
    y_train=nsl_y_train,
    model=nsl_model,
    output_file="results/nsl_kdd_lime_global_importance.csv",
    dataset_name="NSL-KDD"
)


# ============================================================
# 2. RT-IoT2022
# ============================================================

print("\nLoading RT-IoT2022...")

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


# Safety check: leakage feature must NOT exist
if "Unnamed: 0" in rt_X_train.columns:
    raise ValueError(
        "ERROR: 'Unnamed: 0' is still present in RT-IoT2022 features."
    )


# Number of classes
rt_n_classes = rt_y_train.nunique()


# Train the same fixed XGBoost model
rt_model = XGBClassifier(
    **xgb_params,
    objective="multi:softprob",
    num_class=rt_n_classes
)

rt_model.fit(
    rt_X_train,
    rt_y_train
)


rt_lime = generate_lime_importance(
    X_train=rt_X_train,
    X_test=rt_X_test,
    y_train=rt_y_train,
    model=rt_model,
    output_file="results/rt_iot2022_lime_global_importance.csv",
    dataset_name="RT-IoT2022"
)


# ============================================================
# 3. SAVE TOP-10 SUMMARY
# ============================================================

nsl_top10 = nsl_lime.head(10).copy()
nsl_top10["dataset"] = "NSL-KDD"

rt_top10 = rt_lime.head(10).copy()
rt_top10["dataset"] = "RT-IoT2022"

top10_summary = pd.concat(
    [nsl_top10, rt_top10],
    ignore_index=True
)

top10_summary = top10_summary[
    ["dataset", "rank", "feature", "mean_abs_lime"]
]

top10_summary.to_csv(
    "results/milestone_6_lime_top10_summary.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("MILESTONE 6 COMPLETED")
print("=" * 60)

print("\nOutput files:")
print("1. results/nsl_kdd_lime_global_importance.csv")
print("2. results/rt_iot2022_lime_global_importance.csv")
print("3. results/milestone_6_lime_top10_summary.csv")

print("\nLIME protocol:")
print("- LimeTabularExplainer")
print("- 300 fixed random test instances")
print("- Random state = 42")
print("- Global importance = mean(|local weight|)")