# ============================================================
# MILESTONE 8 - E-19 STABILITY AUDIT
# FAST + ROBUST VERSION
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier, DMatrix
from lime.lime_tabular import LimeTabularExplainer

from mealpy.swarm_based import GWO
from mealpy.utils.space import BinaryVar

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42

# Fast audit settings
N_BOOTSTRAPS = 3
N_GWO_RUNS = 3
GWO_EPOCHS = 3
GWO_POP_SIZE = 5

LIME_SAMPLES = 100

# IMPORTANT:
# 10% gives RT-IoT2022 rare classes enough samples
TRAIN_SUBSAMPLE_FRAC = 0.10

VALIDATION_SIZE = 0.20

TOP_K = 10


# ============================================================
# FIXED XGBOOST PARAMETERS
# ============================================================

XGB_PARAMS = {
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
# LOAD DATA
# ============================================================

def load_dataset(dataset_name):

    if dataset_name == "NSL-KDD":

        X_train_path = os.path.join(
            DATA_DIR,
            "nsl_kdd_X_train_encoded.csv"
        )

        X_test_path = os.path.join(
            DATA_DIR,
            "nsl_kdd_X_test_encoded.csv"
        )

        y_train_path = os.path.join(
            DATA_DIR,
            "nsl_kdd_y_train_encoded.csv"
        )

        y_test_path = os.path.join(
            DATA_DIR,
            "nsl_kdd_y_test_encoded.csv"
        )

    else:

        X_train_path = os.path.join(
            DATA_DIR,
            "rt_iot2022_X_train_encoded.csv"
        )

        X_test_path = os.path.join(
            DATA_DIR,
            "rt_iot2022_X_test_encoded.csv"
        )

        y_train_path = os.path.join(
            DATA_DIR,
            "rt_iot2022_y_train_encoded.csv"
        )

        y_test_path = os.path.join(
            DATA_DIR,
            "rt_iot2022_y_test_encoded.csv"
        )

    X_train = pd.read_csv(
        X_train_path
    )

    X_test = pd.read_csv(
        X_test_path
    )

    y_train_df = pd.read_csv(
        y_train_path
    )

    y_test_df = pd.read_csv(
        y_test_path
    )

    y_train = y_train_df.iloc[:, 0]
    y_test = y_test_df.iloc[:, 0]

    # Remove accidental index columns
    unwanted = [
        c for c in X_train.columns
        if c.startswith("Unnamed:")
    ]

    if unwanted:

        X_train = X_train.drop(
            columns=unwanted
        )

        X_test = X_test.drop(
            columns=unwanted,
            errors="ignore"
        )

    print(
        f"Train shape: {X_train.shape}"
    )

    print(
        f"Test shape : {X_test.shape}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# TRAIN XGBOOST
# ============================================================

def train_xgb(
    X_train,
    y_train
):

    n_classes = len(
        np.unique(y_train)
    )

    if n_classes == 2:

        model = XGBClassifier(
            **XGB_PARAMS,
            objective="binary:logistic"
        )

    else:

        model = XGBClassifier(
            **XGB_PARAMS,
            objective="multi:softprob",
            num_class=n_classes
        )

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# SHAP RANKING
# ============================================================

def get_shap_ranking(
    model,
    X,
    feature_names
):

    booster = model.get_booster()

    dmatrix = DMatrix(X)

    contributions = booster.predict(
        dmatrix,
        pred_contribs=True
    )

    if contributions.ndim == 2:

        shap_values = contributions[:, :-1]

        importance = np.mean(
            np.abs(shap_values),
            axis=0
        )

    else:

        shap_values = contributions[:, :, :-1]

        importance = np.mean(
            np.abs(shap_values),
            axis=(0, 1)
        )

    ranking = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    })

    ranking = ranking.sort_values(
        "importance",
        ascending=False
    ).reset_index(drop=True)

    return ranking


# ============================================================
# LIME RANKING
# ============================================================

def get_lime_ranking(
    model,
    X_train,
    X_data,
    feature_names,
    n_samples=LIME_SAMPLES,
    seed=42
):

    rng = np.random.default_rng(seed)

    n_samples = min(
        n_samples,
        len(X_data)
    )

    indices = rng.choice(
        len(X_data),
        size=n_samples,
        replace=False
    )

    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=None,
        mode="classification",
        discretize_continuous=False,
        random_state=seed
    )

    importance_sum = np.zeros(
        len(feature_names),
        dtype=float
    )

    for counter, idx in enumerate(
        indices,
        start=1
    ):

        explanation = explainer.explain_instance(
            X_data[idx],
            model.predict_proba,
            num_features=len(feature_names)
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Directly use local_exp.
        # Do NOT parse as_list().
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            X_data[idx].reshape(1, -1)
        )[0]

        predicted_class = int(
            np.argmax(probabilities)
        )

        if predicted_class in explanation.local_exp:

            local_weights = (
                explanation.local_exp[
                    predicted_class
                ]
            )

        else:

            first_key = list(
                explanation.local_exp.keys()
            )[0]

            local_weights = (
                explanation.local_exp[
                    first_key
                ]
            )

        for feature_idx, weight in local_weights:

            if (
                0 <= feature_idx
                < len(feature_names)
            ):

                importance_sum[
                    feature_idx
                ] += abs(weight)

        if counter % 25 == 0:

            print(
                f"      LIME: "
                f"{counter}/{n_samples}"
            )

    importance = (
        importance_sum
        / n_samples
    )

    ranking = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    })

    ranking = ranking.sort_values(
        "importance",
        ascending=False
    ).reset_index(drop=True)

    return ranking


# ============================================================
# CLASS-AWARE SUBSET
# ============================================================

def create_class_aware_subset(
    X,
    y,
    fraction,
    seed
):
    """
    Creates a subset while preserving every class.

    This fixes the RT-IoT2022 rare-class problem.
    """

    rng = np.random.default_rng(seed)

    y_array = np.asarray(y)

    selected_indices = []

    classes = np.unique(
        y_array
    )

    for cls in classes:

        class_indices = np.where(
            y_array == cls
        )[0]

        n_class = len(
            class_indices
        )

        # ----------------------------------------------------
        # Select fraction of each class
        # At least 2 samples per class when possible.
        # ----------------------------------------------------

        n_select = int(
            np.ceil(
                n_class * fraction
            )
        )

        n_select = max(
            2,
            n_select
        )

        n_select = min(
            n_select,
            n_class
        )

        chosen = rng.choice(
            class_indices,
            size=n_select,
            replace=False
        )

        selected_indices.extend(
            chosen.tolist()
        )

    selected_indices = np.array(
        selected_indices,
        dtype=int
    )

    rng.shuffle(
        selected_indices
    )

    X_subset = X[
        selected_indices
    ]

    if isinstance(y, pd.Series):

        y_subset = y.iloc[
            selected_indices
        ].reset_index(
            drop=True
        )

    else:

        y_subset = y[
            selected_indices
        ]

    return (
        X_subset,
        y_subset
    )


# ============================================================
# SAFE TRAIN / VALIDATION SPLIT
# ============================================================

def safe_train_validation_split(
    X,
    y,
    seed
):
    """
    Guarantees that every class has enough samples.

    The class-aware subset should already contain >=2 samples
    per class.
    """

    y_series = pd.Series(
        y
    ).reset_index(
        drop=True
    )

    X_array = np.asarray(
        X
    )

    class_counts = (
        y_series.value_counts()
    )

    # --------------------------------------------------------
    # If all classes have >=2 samples, stratify.
    # --------------------------------------------------------

    if (
        len(class_counts) > 1
        and class_counts.min() >= 2
    ):

        return train_test_split(
            X_array,
            y_series,
            test_size=VALIDATION_SIZE,
            random_state=seed,
            stratify=y_series
        )

    # --------------------------------------------------------
    # Extremely defensive fallback
    # --------------------------------------------------------

    print(
        "      WARNING: "
        "Stratified split unavailable."
    )

    rng = np.random.default_rng(
        seed
    )

    indices = np.arange(
        len(y_series)
    )

    rng.shuffle(
        indices
    )

    n_val = max(
        1,
        int(
            len(indices)
            * VALIDATION_SIZE
        )
    )

    val_indices = indices[
        :n_val
    ]

    train_indices = indices[
        n_val:
    ]

    return (
        X_array[train_indices],
        X_array[val_indices],
        y_series.iloc[
            train_indices
        ],
        y_series.iloc[
            val_indices
        ]
    )


# ============================================================
# GWO OBJECTIVE
# ============================================================

def make_gwo_objective(
    X_fit,
    y_fit,
    X_val,
    y_val
):

    def objective(solution):

        mask = (
            np.asarray(solution)
            >= 0.5
        )

        # Prevent empty subset
        if mask.sum() == 0:

            return 1.0

        X_fit_selected = (
            X_fit[:, mask]
        )

        X_val_selected = (
            X_val[:, mask]
        )

        try:

            proxy = DecisionTreeClassifier(
                max_depth=5,
                random_state=RANDOM_STATE
            )

            proxy.fit(
                X_fit_selected,
                y_fit
            )

            predictions = proxy.predict(
                X_val_selected
            )

            score = f1_score(
                y_val,
                predictions,
                average="macro"
            )

            return 1.0 - score

        except Exception:

            return 1.0

    return objective


# ============================================================
# GWO RANKING
# ============================================================

def get_gwo_ranking(
    X_train,
    y_train,
    feature_names,
    seed_offset=0
):

    n_features = X_train.shape[1]

    selection_count = np.zeros(
        n_features,
        dtype=int
    )

    for run in range(
        N_GWO_RUNS
    ):

        seed = (
            RANDOM_STATE
            + seed_offset
            + run
        )

        # ----------------------------------------------------
        # CLASS-AWARE SUBSET
        # ----------------------------------------------------

        X_subset, y_subset = (
            create_class_aware_subset(
                X_train,
                y_train,
                TRAIN_SUBSAMPLE_FRAC,
                seed
            )
        )

        # ----------------------------------------------------
        # SAFE VALIDATION SPLIT
        # ----------------------------------------------------

        (
            X_fit,
            X_val,
            y_fit,
            y_val
        ) = safe_train_validation_split(
            X_subset,
            y_subset,
            seed
        )

        objective = make_gwo_objective(
            X_fit,
            y_fit,
            X_val,
            y_val
        )

        problem = {

            "obj_func": objective,

            "bounds": BinaryVar(
                n_vars=n_features,
                name="feature_selection"
            ),

            "minmax": "min"
        }

        optimizer = GWO.OriginalGWO(
            epoch=GWO_EPOCHS,
            pop_size=GWO_POP_SIZE
        )

        print(
            f"      GWO run "
            f"{run + 1}/{N_GWO_RUNS}"
        )

        solution = optimizer.solve(
            problem
        )

        position = np.asarray(
            solution.solution
        )

        mask = (
            position >= 0.5
        )

        # ----------------------------------------------------
        # Safety fallback
        # ----------------------------------------------------

        if mask.sum() == 0:

            best_idx = np.argmax(
                position
            )

            mask[
                best_idx
            ] = True

        selection_count += (
            mask.astype(int)
        )

    frequency = (
        selection_count
        / N_GWO_RUNS
    )

    ranking = pd.DataFrame({

        "feature":
            feature_names,

        "selection_frequency":
            frequency

    })

    ranking = ranking.sort_values(
        "selection_frequency",
        ascending=False
    ).reset_index(drop=True)

    return ranking


# ============================================================
# TOP-K
# ============================================================

def get_top_features(
    ranking,
    k=TOP_K
):

    return ranking.head(k)[
        "feature"
    ].tolist()


# ============================================================
# JACCARD
# ============================================================

def jaccard_similarity(
    set_a,
    set_b
):

    a = set(
        set_a
    )

    b = set(
        set_b
    )

    union = a | b

    if len(union) == 0:

        return 1.0

    return len(
        a & b
    ) / len(union)


# ============================================================
# BOOTSTRAP DATA
# ============================================================

def bootstrap_sample(
    X,
    y,
    seed
):

    rng = np.random.default_rng(
        seed
    )

    indices = rng.choice(
        len(X),
        size=len(X),
        replace=True
    )

    X_boot = X[
        indices
    ]

    if isinstance(y, pd.Series):

        y_boot = y.iloc[
            indices
        ].reset_index(
            drop=True
        )

    else:

        y_boot = y[
            indices
        ]

    return (
        X_boot,
        y_boot
    )


# ============================================================
# ONE DATASET STABILITY
# ============================================================

def run_stability(
    dataset_name,
    X_train,
    X_test,
    y_train,
    y_test
):

    print()
    print("=" * 60)
    print(
        f"E-19 STABILITY — "
        f"{dataset_name}"
    )
    print("=" * 60)

    print(
        f"Train shape: {X_train.shape}"
    )

    print(
        f"Test shape : {X_test.shape}"
    )

    # Convert to numpy
    X_train_np = np.asarray(
        X_train,
        dtype=np.float64
    )

    X_test_np = np.asarray(
        X_test,
        dtype=np.float64
    )

    y_train = pd.Series(
        y_train
    ).reset_index(
        drop=True
    )

    y_test = pd.Series(
        y_test
    ).reset_index(
        drop=True
    )

    feature_names = list(
        X_train.columns
        if isinstance(
            X_train,
            pd.DataFrame
        )
        else [
            f"feature_{i}"
            for i in range(
                X_train_np.shape[1]
            )
        ]
    )

    # ========================================================
    # FIXED XGBOOST
    # ========================================================

    print()
    print(
        "Training fixed XGBoost model..."
    )

    model = train_xgb(
        X_train_np,
        y_train
    )

    # ========================================================
    # ORIGINAL SHAP
    # ========================================================

    print()
    print(
        "Generating original SHAP ranking..."
    )

    shap_original = (
        get_shap_ranking(
            model,
            X_test_np,
            feature_names
        )
    )

    shap_top = get_top_features(
        shap_original
    )

    print(
        "Original SHAP Top-10:"
    )

    print(
        shap_top
    )

    # ========================================================
    # ORIGINAL LIME
    # ========================================================

    print()
    print(
        "Generating original LIME ranking..."
    )

    lime_original = (
        get_lime_ranking(
            model,
            X_train_np,
            X_test_np,
            feature_names,
            LIME_SAMPLES,
            RANDOM_STATE
        )
    )

    lime_top = get_top_features(
        lime_original
    )

    print(
        "Original LIME Top-10:"
    )

    print(
        lime_top
    )

    # ========================================================
    # ORIGINAL GWO
    # ========================================================

    print()
    print(
        "Generating original GWO ranking..."
    )

    gwo_original = (
        get_gwo_ranking(
            X_train_np,
            y_train,
            feature_names,
            seed_offset=0
        )
    )

    gwo_top = get_top_features(
        gwo_original
    )

    print(
        "Original GWO Top-10:"
    )

    print(
        gwo_top
    )

    # ========================================================
    # BOOTSTRAP RESULTS
    # ========================================================

    results = []

    for bootstrap_id in range(
        1,
        N_BOOTSTRAPS + 1
    ):

        print()
        print(
            f"Bootstrap "
            f"{bootstrap_id}/"
            f"{N_BOOTSTRAPS}"
        )

        bootstrap_seed = (
            RANDOM_STATE
            + bootstrap_id
            * 100
        )

        (
            X_boot,
            y_boot
        ) = bootstrap_sample(
            X_train_np,
            y_train,
            bootstrap_seed
        )

        # ====================================================
        # SHAP
        # ====================================================

        print(
            "  SHAP..."
        )

        shap_boot = (
            get_shap_ranking(
                model,
                X_test_np,
                feature_names
            )
        )

        shap_boot_top = (
            get_top_features(
                shap_boot
            )
        )

        shap_jaccard = (
            jaccard_similarity(
                shap_top,
                shap_boot_top
            )
        )

        # ====================================================
        # LIME
        # ====================================================

        print(
            "  LIME..."
        )

        lime_boot = (
            get_lime_ranking(
                model,
                X_boot,
                X_test_np,
                feature_names,
                LIME_SAMPLES,
                bootstrap_seed
            )
        )

        lime_boot_top = (
            get_top_features(
                lime_boot
            )
        )

        lime_jaccard = (
            jaccard_similarity(
                lime_top,
                lime_boot_top
            )
        )

        # ====================================================
        # GWO
        # ====================================================

        print(
            "  GWO..."
        )

        gwo_boot = (
            get_gwo_ranking(
                X_boot,
                y_boot,
                feature_names,
                seed_offset=bootstrap_id
                * 1000
            )
        )

        gwo_boot_top = (
            get_top_features(
                gwo_boot
            )
        )

        gwo_jaccard = (
            jaccard_similarity(
                gwo_top,
                gwo_boot_top
            )
        )

        # ====================================================
        # PRINT
        # ====================================================

        print(
            f"  SHAP Jaccard = "
            f"{shap_jaccard:.4f}"
        )

        print(
            f"  LIME Jaccard = "
            f"{lime_jaccard:.4f}"
        )

        print(
            f"  GWO Jaccard  = "
            f"{gwo_jaccard:.4f}"
        )

        # ====================================================
        # SAVE
        # ====================================================

        results.append({

            "dataset":
                dataset_name,

            "bootstrap":
                bootstrap_id,

            "method":
                "SHAP",

            "top_k":
                TOP_K,

            "jaccard_similarity":
                shap_jaccard
        })

        results.append({

            "dataset":
                dataset_name,

            "bootstrap":
                bootstrap_id,

            "method":
                "LIME",

            "top_k":
                TOP_K,

            "jaccard_similarity":
                lime_jaccard
        })

        results.append({

            "dataset":
                dataset_name,

            "bootstrap":
                bootstrap_id,

            "method":
                "GWO",

            "top_k":
                TOP_K,

            "jaccard_similarity":
                gwo_jaccard
        })

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    all_results = []

    # ========================================================
    # NSL-KDD
    # ========================================================

    (
        X_train_nsl,
        X_test_nsl,
        y_train_nsl,
        y_test_nsl
    ) = load_dataset(
        "NSL-KDD"
    )

    nsl_results = run_stability(
        "NSL-KDD",
        X_train_nsl,
        X_test_nsl,
        y_train_nsl,
        y_test_nsl
    )

    all_results.extend(
        nsl_results
    )

    # ========================================================
    # RT-IoT2022
    # ========================================================

    (
        X_train_rt,
        X_test_rt,
        y_train_rt,
        y_test_rt
    ) = load_dataset(
        "RT-IoT2022"
    )

    rt_results = run_stability(
        "RT-IoT2022",
        X_train_rt,
        X_test_rt,
        y_train_rt,
        y_test_rt
    )

    all_results.extend(
        rt_results
    )

    # ========================================================
    # SAVE
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    output_path = os.path.join(
        RESULTS_DIR,
        "milestone_8_stability.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = (
        results_df
        .groupby(
            [
                "dataset",
                "method"
            ]
        )[
            "jaccard_similarity"
        ]
        .agg(
            [
                "mean",
                "std"
            ]
        )
        .reset_index()
    )

    summary = summary.rename(
        columns={
            "mean":
                "mean_jaccard",

            "std":
                "std_jaccard"
        }
    )

    summary_path = os.path.join(
        RESULTS_DIR,
        "milestone_8_stability_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 60)
    print(
        "E-19 STABILITY RESULTS"
    )
    print("=" * 60)

    print()

    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print(
        "SUMMARY"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved:"
    )

    print(
        output_path
    )

    print(
        summary_path
    )

    print()
    print("=" * 60)
    print(
        "E-19 STABILITY COMPLETED"
    )
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()