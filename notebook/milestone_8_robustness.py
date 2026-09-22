# ============================================================
# MILESTONE 8 - E20 ROBUSTNESS AUDIT
# Fast + Robust Version
# ============================================================

import os
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier, DMatrix
from lime.lime_tabular import LimeTabularExplainer

from mealpy.swarm_based import GWO
from mealpy.utils.space import BinaryVar

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

DATA_DIR = "data"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# Fast settings
LIME_SAMPLES = 100

N_GWO_RUNS = 3
GWO_EPOCHS = 3
GWO_POP_SIZE = 5

# Use 10% instead of 5% for RT-IoT2022
# This helps preserve rare classes.
TRAIN_SUBSAMPLE_FRAC = 0.10

VALIDATION_SIZE = 0.20

# Noise levels required by the project
SIGMAS = [0.1, 0.3]

# ============================================================
# XGBOOST PARAMETERS
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
# HELPER FUNCTIONS
# ============================================================

def train_xgb(X_train, y_train, n_classes):
    """
    Train the fixed XGBoost classifier.
    """

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

    model.fit(X_train, y_train)

    return model


# ============================================================

def macro_f1(model, X, y):
    """
    Calculate Macro-F1.
    """

    predictions = model.predict(X)

    return f1_score(
        y,
        predictions,
        average="macro"
    )


# ============================================================

def get_shap_ranking(model, X_test, feature_names):
    """
    Native XGBoost TreeSHAP.

    This avoids the SHAP/XGBoost 3.2 compatibility problem.
    """

    booster = model.get_booster()

    dmatrix = DMatrix(X_test)

    contributions = booster.predict(
        dmatrix,
        pred_contribs=True
    )

    # --------------------------------------------------------
    # Binary classification
    # --------------------------------------------------------

    if contributions.ndim == 2:

        shap_values = contributions[:, :-1]

        importance = np.mean(
            np.abs(shap_values),
            axis=0
        )

    # --------------------------------------------------------
    # Multiclass classification
    # --------------------------------------------------------

    else:

        shap_values = contributions[:, :, :-1]

        importance = np.mean(
            np.abs(shap_values),
            axis=(0, 1)
        )

    ranking = (
        pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        })
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return ranking


# ============================================================

def get_lime_ranking(
    model,
    X_train,
    X_test,
    feature_names,
    n_samples=100
):
    """
    Calculate global LIME importance by averaging
    absolute local explanation weights.
    """

    rng = np.random.default_rng(RANDOM_STATE)

    n_samples = min(
        n_samples,
        len(X_test)
    )

    sample_indices = rng.choice(
        len(X_test),
        size=n_samples,
        replace=False
    )

    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=None,
        mode="classification",
        discretize_continuous=False,
        random_state=RANDOM_STATE
    )

    importance_sum = np.zeros(
        len(feature_names),
        dtype=float
    )

    count = 0

    for counter, idx in enumerate(sample_indices, start=1):

        explanation = explainer.explain_instance(
            X_test[idx],
            model.predict_proba,
            num_features=len(feature_names)
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Use local_exp directly instead of parsing as_list()
        # ----------------------------------------------------

        local_exp = explanation.local_exp

        # Get explanation for predicted class
        predicted_class = int(
            np.argmax(
                model.predict_proba(
                    X_test[idx].reshape(1, -1)
                )[0]
            )
        )

        if predicted_class in local_exp:

            weights = local_exp[predicted_class]

        else:

            # fallback
            first_key = list(local_exp.keys())[0]
            weights = local_exp[first_key]

        for feature_idx, weight in weights:

            if 0 <= feature_idx < len(feature_names):

                importance_sum[feature_idx] += abs(weight)

        count += 1

        if counter % 25 == 0:
            print(
                f"      LIME: {counter}/{n_samples}"
            )

    importance = importance_sum / max(count, 1)

    ranking = (
        pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        })
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return ranking


# ============================================================

def create_class_aware_subset(
    X,
    y,
    fraction,
    seed
):
    """
    Create a class-aware subset.

    This prevents the RT-IoT2022 rare-class problem.

    Every class gets at least 2 samples whenever possible.
    """

    rng = np.random.default_rng(seed)

    selected_indices = []

    y_array = np.asarray(y)

    unique_classes = np.unique(y_array)

    for cls in unique_classes:

        class_indices = np.where(
            y_array == cls
        )[0]

        n_class = len(class_indices)

        # At least 2 samples per class when possible
        n_select = max(
            2,
            int(np.ceil(
                n_class * fraction
            ))
        )

        n_select = min(
            n_select,
            n_class
        )

        selected = rng.choice(
            class_indices,
            size=n_select,
            replace=False
        )

        selected_indices.extend(
            selected.tolist()
        )

    rng.shuffle(selected_indices)

    selected_indices = np.array(
        selected_indices,
        dtype=int
    )

    return (
        X[selected_indices],
        y.iloc[selected_indices]
        if isinstance(y, pd.Series)
        else y_array[selected_indices]
    )


# ============================================================

def gwo_objective_factory(
    X_fit,
    y_fit,
    X_val,
    y_val,
    n_features
):
    """
    Create GWO objective function.

    A Decision Tree proxy classifier is used for speed.
    """

    def objective(solution):

        mask = np.asarray(solution) >= 0.5

        # Avoid empty feature subset
        if mask.sum() == 0:

            return 1.0

        X_fit_selected = X_fit[:, mask]
        X_val_selected = X_val[:, mask]

        proxy = DecisionTreeClassifier(
            max_depth=5,
            random_state=RANDOM_STATE
        )

        try:

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

def run_gwo(
    X_train,
    y_train,
    feature_names,
    n_runs=N_GWO_RUNS,
    epochs=GWO_EPOCHS,
    pop_size=GWO_POP_SIZE,
    fraction=TRAIN_SUBSAMPLE_FRAC
):
    """
    Fast GWO feature selection.

    Uses class-aware subset sampling and one validation split.
    """

    n_features = X_train.shape[1]

    selection_count = np.zeros(
        n_features,
        dtype=int
    )

    print(
        f"      GWO: {n_runs} runs | "
        f"{epochs} epochs | "
        f"population={pop_size}"
    )

    for run in range(n_runs):

        seed = RANDOM_STATE + run

        # ----------------------------------------------------
        # Class-aware subset
        # ----------------------------------------------------

        X_sub, y_sub = create_class_aware_subset(
            X_train,
            y_train,
            fraction,
            seed
        )

        # ----------------------------------------------------
        # Safe validation split
        # ----------------------------------------------------

        class_counts = (
            pd.Series(y_sub)
            .value_counts()
        )

        if class_counts.min() >= 2:

            stratify_value = y_sub

        else:

            stratify_value = None

        X_fit, X_val, y_fit, y_val = train_test_split(
            X_sub,
            y_sub,
            test_size=VALIDATION_SIZE,
            random_state=seed,
            stratify=stratify_value
        )

        objective = gwo_objective_factory(
            X_fit,
            y_fit,
            X_val,
            y_val,
            n_features
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
            epoch=epochs,
            pop_size=pop_size
        )

        print(
            f"      GWO run {run + 1}/{n_runs}"
        )

        start_time = time.time()

        try:

            solution = optimizer.solve(
                problem
            )

            position = np.asarray(
                solution.solution
            )

            mask = position >= 0.5

            # ------------------------------------------------
            # Safety fallback
            # ------------------------------------------------

            if mask.sum() == 0:

                best_idx = np.argmax(
                    position
                )

                mask[best_idx] = True

            selection_count += mask.astype(int)

            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"        selected={mask.sum()} "
                f"features | "
                f"time={elapsed:.2f}s"
            )

        except Exception as e:

            print(
                f"        GWO run failed: {e}"
            )

    # --------------------------------------------------------
    # Selection frequency
    # --------------------------------------------------------

    frequency = (
        selection_count / n_runs
    )

    ranking = (
        pd.DataFrame({
            "feature": feature_names,
            "selection_frequency": frequency
        })
        .sort_values(
            "selection_frequency",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return ranking


# ============================================================

def add_feature_scaled_noise(
    X,
    sigma,
    seed
):
    """
    Feature-wise Gaussian noise.

    Noise standard deviation for feature j:

        sigma * std(feature_j)

    This is the required robustness protocol.
    """

    rng = np.random.default_rng(seed)

    X_float = np.asarray(
        X,
        dtype=np.float64
    )

    feature_std = np.std(
        X_float,
        axis=0
    )

    # Avoid zero-standard-deviation issues
    feature_std[
        feature_std == 0
    ] = 1.0

    noise = rng.normal(
        loc=0.0,
        scale=sigma * feature_std,
        size=X_float.shape
    )

    noisy_X = X_float + noise

    return noisy_X


# ============================================================

def top_k_features(ranking, k):
    """
    Return top-k feature names.
    """

    return ranking.head(k)[
        "feature"
    ].tolist()


# ============================================================

def get_feature_indices(
    feature_names,
    selected_features
):

    feature_to_index = {
        feature: idx
        for idx, feature
        in enumerate(feature_names)
    }

    return [
        feature_to_index[f]
        for f in selected_features
        if f in feature_to_index
    ]


# ============================================================

def jaccard_similarity(
    original_features,
    noisy_features
):
    """
    Jaccard similarity between top-k feature sets.
    """

    set_a = set(
        original_features
    )

    set_b = set(
        noisy_features
    )

    union = set_a | set_b

    if len(union) == 0:
        return 1.0

    return len(
        set_a & set_b
    ) / len(union)


# ============================================================

def evaluate_robustness(
    dataset_name,
    X_train,
    y_train,
    X_test,
    y_test
):

    print()
    print("=" * 70)
    print(
        f"ROBUSTNESS: {dataset_name}"
    )
    print("=" * 70)

    feature_names = list(
        X_train.columns
        if isinstance(X_train, pd.DataFrame)
        else [
            f"feature_{i}"
            for i in range(X_train.shape[1])
        ]
    )

    # Convert to numpy float
    X_train_np = np.asarray(
        X_train,
        dtype=np.float64
    )

    X_test_np = np.asarray(
        X_test,
        dtype=np.float64
    )

    # ========================================================
    # TRAIN BASELINE XGBOOST
    # ========================================================

    print("\n[1/5] Training XGBoost...")

    n_classes = len(
        np.unique(y_train)
    )

    model = train_xgb(
        X_train_np,
        y_train,
        n_classes
    )

    baseline_f1 = macro_f1(
        model,
        X_test_np,
        y_test
    )

    print(
        f"Baseline Macro-F1: "
        f"{baseline_f1:.6f}"
    )

    # ========================================================
    # ORIGINAL SHAP
    # ========================================================

    print("\n[2/5] Calculating original SHAP ranking...")

    shap_original = get_shap_ranking(
        model,
        X_test_np,
        feature_names
    )

    # ========================================================
    # ORIGINAL LIME
    # ========================================================

    print("\n[3/5] Calculating original LIME ranking...")

    lime_original = get_lime_ranking(
        model,
        X_train_np,
        X_test_np,
        feature_names,
        LIME_SAMPLES
    )

    # ========================================================
    # ORIGINAL GWO
    # ========================================================

    print("\n[4/5] Calculating original GWO ranking...")

    gwo_original = run_gwo(
        X_train_np,
        np.asarray(y_train),
        feature_names
    )

    # ========================================================
    # TOP-K
    # ========================================================

    # Use top 10 because robustness compares ranking changes
    TOP_K = min(
        10,
        len(feature_names)
    )

    original_shap_top = top_k_features(
        shap_original,
        TOP_K
    )

    original_lime_top = top_k_features(
        lime_original,
        TOP_K
    )

    original_gwo_top = top_k_features(
        gwo_original,
        TOP_K
    )

    # ========================================================
    # RESULTS
    # ========================================================

    results = []

    # ========================================================
    # NOISE EXPERIMENTS
    # ========================================================

    print("\n[5/5] Running noise robustness...")

    for sigma in SIGMAS:

        print()
        print(
            f"--- Noise sigma = {sigma} ---"
        )

        # ----------------------------------------------------
        # NOISY TEST DATA
        # ----------------------------------------------------

        X_test_noisy = add_feature_scaled_noise(
            X_test_np,
            sigma,
            RANDOM_STATE
        )

        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        print("  SHAP...")

        shap_noisy = get_shap_ranking(
            model,
            X_test_noisy,
            feature_names
        )

        noisy_shap_top = top_k_features(
            shap_noisy,
            TOP_K
        )

        shap_jaccard = jaccard_similarity(
            original_shap_top,
            noisy_shap_top
        )

        # ----------------------------------------------------
        # LIME
        # ----------------------------------------------------

        print("  LIME...")

        lime_noisy = get_lime_ranking(
            model,
            X_train_np,
            X_test_noisy,
            feature_names,
            LIME_SAMPLES
        )

        noisy_lime_top = top_k_features(
            lime_noisy,
            TOP_K
        )

        lime_jaccard = jaccard_similarity(
            original_lime_top,
            noisy_lime_top
        )

        # ----------------------------------------------------
        # GWO
        # ----------------------------------------------------

        print("  GWO...")

        # Noise is applied to TRAINING data for GWO
        X_train_noisy = add_feature_scaled_noise(
            X_train_np,
            sigma,
            RANDOM_STATE + 1000
        )

        gwo_noisy = run_gwo(
            X_train_noisy,
            np.asarray(y_train),
            feature_names
        )

        noisy_gwo_top = top_k_features(
            gwo_noisy,
            TOP_K
        )

        gwo_jaccard = jaccard_similarity(
            original_gwo_top,
            noisy_gwo_top
        )

        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print(
            f"  SHAP Jaccard: "
            f"{shap_jaccard:.4f}"
        )

        print(
            f"  LIME Jaccard: "
            f"{lime_jaccard:.4f}"
        )

        print(
            f"  GWO Jaccard: "
            f"{gwo_jaccard:.4f}"
        )

        # ----------------------------------------------------
        # SAVE RESULTS
        # ----------------------------------------------------

        results.append({
            "dataset": dataset_name,
            "sigma": sigma,
            "method": "SHAP",
            "top_k": TOP_K,
            "jaccard_similarity": shap_jaccard
        })

        results.append({
            "dataset": dataset_name,
            "sigma": sigma,
            "method": "LIME",
            "top_k": TOP_K,
            "jaccard_similarity": lime_jaccard
        })

        results.append({
            "dataset": dataset_name,
            "sigma": sigma,
            "method": "GWO",
            "top_k": TOP_K,
            "jaccard_similarity": gwo_jaccard
        })

    return results


# ============================================================
# DATASET LOADER
# ============================================================

def load_dataset_files(
    dataset_name
):

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

    print()
    print(
        f"Loading {dataset_name}..."
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

    # --------------------------------------------------------
    # Convert target dataframe to Series
    # --------------------------------------------------------

    if y_train_df.shape[1] == 1:

        y_train = y_train_df.iloc[:, 0]

    else:

        y_train = y_train_df.squeeze()

    if y_test_df.shape[1] == 1:

        y_test = y_test_df.iloc[:, 0]

    else:

        y_test = y_test_df.squeeze()

    # --------------------------------------------------------
    # Remove accidental index columns
    # --------------------------------------------------------

    unwanted_columns = [
        col
        for col in X_train.columns
        if col.startswith("Unnamed:")
    ]

    if unwanted_columns:

        print(
            "Removing unwanted columns:",
            unwanted_columns
        )

        X_train = X_train.drop(
            columns=unwanted_columns
        )

        X_test = X_test.drop(
            columns=unwanted_columns,
            errors="ignore"
        )

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "X_test:",
        X_test.shape
    )

    print(
        "Number of classes:",
        y_train.nunique()
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("MILESTONE 8 - E20 ROBUSTNESS AUDIT")
    print("=" * 70)

    all_results = []

    # ========================================================
    # NSL-KDD
    # ========================================================

    (
        X_train_nsl,
        X_test_nsl,
        y_train_nsl,
        y_test_nsl
    ) = load_dataset_files(
        "NSL-KDD"
    )

    nsl_results = evaluate_robustness(
        "NSL-KDD",
        X_train_nsl,
        y_train_nsl,
        X_test_nsl,
        y_test_nsl
    )

    all_results.extend(
        nsl_results
    )

    # ========================================================
    # RT-IOT2022
    # ========================================================

    (
        X_train_rt,
        X_test_rt,
        y_train_rt,
        y_test_rt
    ) = load_dataset_files(
        "RT-IoT2022"
    )

    rt_results = evaluate_robustness(
        "RT-IoT2022",
        X_train_rt,
        y_train_rt,
        X_test_rt,
        y_test_rt
    )

    all_results.extend(
        rt_results
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_path = os.path.join(
        RESULTS_DIR,
        "milestone_8_robustness.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = (
        results_df
        .pivot_table(
            index=[
                "dataset",
                "sigma"
            ],
            columns="method",
            values="jaccard_similarity"
        )
        .reset_index()
    )

    summary_path = os.path.join(
        RESULTS_DIR,
        "milestone_8_robustness_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("ROBUSTNESS RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved:",
        results_path
    )

    print(
        "Saved:",
        summary_path
    )

    print()
    print("=" * 70)
    print("MILESTONE 8 E20 COMPLETED")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()