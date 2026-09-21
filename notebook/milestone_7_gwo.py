import pandas as pd
import numpy as np

from mealpy.swarm_based import GWO
from mealpy.utils.space import BinaryVar

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


# ============================================================
# MILESTONE 7 — GREY WOLF OPTIMIZER
# Task-General Feature Selection
# ============================================================

RANDOM_STATE = 42

N_RUNS = 30

TRAIN_SUBSAMPLE_FRAC = 0.20

CV_FOLDS = 5

EPOCHS = 20

POP_SIZE = 20


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(dataset_name):

    if dataset_name == "NSL-KDD":

        X_train = pd.read_csv(
            "data/nsl_kdd_X_train_encoded.csv"
        )

        X_test = pd.read_csv(
            "data/nsl_kdd_X_test_encoded.csv"
        )

        y_train = pd.read_csv(
            "data/nsl_kdd_y_train_encoded.csv"
        ).squeeze()

        y_test = pd.read_csv(
            "data/nsl_kdd_y_test_encoded.csv"
        ).squeeze()

    elif dataset_name == "RT-IoT2022":

        X_train = pd.read_csv(
            "data/rt_iot2022_X_train_encoded.csv"
        )

        X_test = pd.read_csv(
            "data/rt_iot2022_X_test_encoded.csv"
        )

        y_train = pd.read_csv(
            "data/rt_iot2022_y_train_encoded.csv"
        ).squeeze()

        y_test = pd.read_csv(
            "data/rt_iot2022_y_test_encoded.csv"
        ).squeeze()

        # Leakage check
        if "Unnamed: 0" in X_train.columns:
            raise ValueError(
                "ERROR: 'Unnamed: 0' is still present "
                "in RT-IoT2022 training features."
            )

        if "Unnamed: 0" in X_test.columns:
            raise ValueError(
                "ERROR: 'Unnamed: 0' is still present "
                "in RT-IoT2022 test features."
            )

    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}"
        )

    return X_train, X_test, y_train, y_test


# ============================================================
# GWO FEATURE SELECTION
# ============================================================

def generate_gwo_importance(
    X_train,
    X_test,
    y_train,
    y_test,
    dataset_name,
    output_file
):

    print("\n" + "=" * 70)
    print(
        f"MILESTONE 7 — {dataset_name} GWO"
    )
    print("=" * 70)

    feature_names = list(
        X_train.columns
    )

    n_features = len(
        feature_names
    )

    print(
        f"Number of features : {n_features}"
    )

    # --------------------------------------------------------
    # Fixed 20% training subsample
    # --------------------------------------------------------

    X_train_sub = X_train.sample(
        frac=TRAIN_SUBSAMPLE_FRAC,
        random_state=RANDOM_STATE
    )

    y_train_sub = y_train.loc[
        X_train_sub.index
    ]

    print(
        f"Training samples     : {len(X_train)}"
    )

    print(
        f"GWO training subset  : {len(X_train_sub)}"
    )

    print(
        f"CV folds             : {CV_FOLDS}"
    )

    print(
        f"Independent GWO runs : {N_RUNS}"
    )

    print(
        f"Epochs per run       : {EPOCHS}"
    )

    print(
        f"Population size      : {POP_SIZE}"
    )

    # ========================================================
    # SELECTION COUNTER
    # ========================================================

    selection_counts = np.zeros(
        n_features,
        dtype=int
    )

    # ========================================================
    # 30 INDEPENDENT GWO RUNS
    # ========================================================

    for run in range(N_RUNS):

        print("\n" + "-" * 60)

        print(
            f"GWO RUN {run + 1}/{N_RUNS}"
        )

        print("-" * 60)

        run_seed = (
            RANDOM_STATE + run
        )

        # ----------------------------------------------------
        # Fitness function
        # ----------------------------------------------------

        def objective(solution):

            mask = (
                np.asarray(solution) >= 0.5
            )

            selected_indices = np.where(
                mask
            )[0]

            # No features selected
            if len(selected_indices) == 0:
                return 1.0

            X_selected = X_train_sub.iloc[
                :,
                selected_indices
            ]

            # ------------------------------------------------
            # Fresh proxy classifier
            # ------------------------------------------------

            clf = DecisionTreeClassifier(
                max_depth=5,
                random_state=RANDOM_STATE
            )

            # ------------------------------------------------
            # 5-fold CV Macro-F1
            # ------------------------------------------------

            scores = cross_val_score(
                clf,
                X_selected,
                y_train_sub,
                cv=CV_FOLDS,
                scoring="f1_macro",
                n_jobs=-1
            )

            mean_macro_f1 = scores.mean()

            # Minimize 1 - Macro-F1
            return 1.0 - mean_macro_f1

        # ----------------------------------------------------
        # MEALPY problem
        # ----------------------------------------------------

        problem = {
            "obj_func": objective,

            "bounds": BinaryVar(
                n_vars=n_features,
                name="feature_selection"
            ),

            "minmax": "min"
        }

        # ----------------------------------------------------
        # GWO optimizer
        # ----------------------------------------------------

        optimizer = GWO.OriginalGWO(
            epoch=EPOCHS,
            pop_size=POP_SIZE
        )

        # Independent seed
        np.random.seed(
            run_seed
        )

        # ----------------------------------------------------
        # Run GWO
        # ----------------------------------------------------

        best_agent = optimizer.solve(
            problem
        )

        # ----------------------------------------------------
        # Extract selected features
        # ----------------------------------------------------

        best_solution = np.asarray(
            best_agent.solution
        )

        selected_mask = (
            best_solution >= 0.5
        )

        selected_indices = np.where(
            selected_mask
        )[0]

        # Safety fallback
        if len(selected_indices) == 0:

            selected_indices = np.array([
                np.argmax(
                    best_solution
                )
            ])

        # ----------------------------------------------------
        # Count feature selections
        # ----------------------------------------------------

        selection_counts[
            selected_indices
        ] += 1

        print(
            f"Selected features: "
            f"{len(selected_indices)}/{n_features}"
        )

    # ========================================================
    # GLOBAL IMPORTANCE
    # ========================================================

    selection_frequency = (
        selection_counts / N_RUNS
    )

    gwo_importance = pd.DataFrame({

        "feature": feature_names,

        "selection_count":
            selection_counts,

        "selection_frequency":
            selection_frequency
    })

    # --------------------------------------------------------
    # Rank by selection frequency
    # --------------------------------------------------------

    gwo_importance = (
        gwo_importance
        .sort_values(
            by=[
                "selection_frequency",
                "selection_count"
            ],
            ascending=False
        )
        .reset_index(drop=True)
    )

    gwo_importance["rank"] = (
        gwo_importance.index + 1
    )

    gwo_importance = gwo_importance[
        [
            "rank",
            "feature",
            "selection_count",
            "selection_frequency"
        ]
    ]

    # ========================================================
    # SAVE FULL RANKING
    # ========================================================

    gwo_importance.to_csv(
        output_file,
        index=False
    )

    print("\nTop 10 GWO features:")

    print(
        gwo_importance
        .head(10)
        .to_string(index=False)
    )

    print(
        f"\nSaved: {output_file}"
    )

    return gwo_importance


# ============================================================
# NSL-KDD
# ============================================================

print(
    "\nLoading NSL-KDD..."
)

(
    nsl_X_train,
    nsl_X_test,
    nsl_y_train,
    nsl_y_test
) = load_dataset(
    "NSL-KDD"
)

nsl_gwo = generate_gwo_importance(
    X_train=nsl_X_train,
    X_test=nsl_X_test,
    y_train=nsl_y_train,
    y_test=nsl_y_test,
    dataset_name="NSL-KDD",
    output_file=(
        "results/"
        "nsl_kdd_gwo_global_importance.csv"
    )
)


# ============================================================
# RT-IoT2022
# ============================================================

print(
    "\nLoading RT-IoT2022..."
)

(
    rt_X_train,
    rt_X_test,
    rt_y_train,
    rt_y_test
) = load_dataset(
    "RT-IoT2022"
)

rt_gwo = generate_gwo_importance(
    X_train=rt_X_train,
    X_test=rt_X_test,
    y_train=rt_y_train,
    y_test=rt_y_test,
    dataset_name="RT-IoT2022",
    output_file=(
        "results/"
        "rt_iot2022_gwo_global_importance.csv"
    )
)


# ============================================================
# TOP-10 SUMMARY
# ============================================================

nsl_top10 = (
    nsl_gwo
    .head(10)
    .copy()
)

nsl_top10["dataset"] = (
    "NSL-KDD"
)

rt_top10 = (
    rt_gwo
    .head(10)
    .copy()
)

rt_top10["dataset"] = (
    "RT-IoT2022"
)

top10_summary = pd.concat(
    [
        nsl_top10,
        rt_top10
    ],
    ignore_index=True
)

top10_summary = top10_summary[
    [
        "dataset",
        "rank",
        "feature",
        "selection_count",
        "selection_frequency"
    ]
]

top10_summary.to_csv(
    "results/"
    "milestone_7_gwo_top10_summary.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "MILESTONE 7 COMPLETED"
)

print(
    "=" * 70
)

print(
    "\nGWO protocol:"
)

print(
    "- 30 independent GWO runs"
)

print(
    "- Fixed 20% training subsample"
)

print(
    "- 5-fold cross-validation"
)

print(
    "- Fresh DecisionTreeClassifier(max_depth=5)"
)

print(
    "- Macro-F1 fitness"
)

print(
    "- No test data used in fitness"
)

print(
    "- Global importance = selection frequency"
)

print(
    "\nOutput files:"
)

print(
    "1. results/nsl_kdd_gwo_global_importance.csv"
)

print(
    "2. results/rt_iot2022_gwo_global_importance.csv"
)

print(
    "3. results/milestone_7_gwo_top10_summary.csv"
)