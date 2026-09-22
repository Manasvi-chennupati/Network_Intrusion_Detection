"""
Milestone 9: Final XAI Audit Comparison

Combines the four Milestone 8 audit dimensions:
1. Deletion Faithfulness
2. Sparsity
3. Stability
4. Robustness

No model training is performed in this milestone.
All values are read directly from Milestone 8 result files.
"""

from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


DELETION_FILE = RESULTS_DIR / "milestone_8_deletion_faithfulness.csv"
SPARSITY_FILE = RESULTS_DIR / "milestone_8_sparsity.csv"
STABILITY_FILE = RESULTS_DIR / "milestone_8_stability_summary.csv"
ROBUSTNESS_FILE = RESULTS_DIR / "milestone_8_robustness.csv"


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = {
    "deletion": [
        "dataset",
        "method",
        "removed_features",
        "test_macro_f1",
        "f1_drop",
    ],
    "sparsity": [
        "dataset",
        "method",
        "baseline_macro_f1",
        "threshold_90_percent",
        "sparsity_k",
        "macro_f1_at_sparsity_k",
    ],
    "stability": [
        "dataset",
        "method",
        "mean_jaccard",
        "std_jaccard",
    ],
    "robustness": [
        "dataset",
        "sigma",
        "method",
        "top_k",
        "jaccard_similarity",
    ],
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def check_columns(df, required, name):
    """Verify that a CSV contains the expected columns."""
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def normalize_method_names(df):
    """Normalize method names for consistent merging."""
    df = df.copy()

    df["method"] = (
        df["method"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return df


def normalize_dataset_names(df):
    """Normalize dataset names for consistent merging."""
    df = df.copy()

    df["dataset"] = (
        df["dataset"]
        .astype(str)
        .str.strip()
    )

    return df


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("MILESTONE 9 — FINAL XAI AUDIT COMPARISON")
print("=" * 70)

print("\nLoading Milestone 8 results...")


# -------------------------
# Deletion faithfulness
# -------------------------

deletion = pd.read_csv(DELETION_FILE)

check_columns(
    deletion,
    REQUIRED_COLUMNS["deletion"],
    "milestone_8_deletion_faithfulness.csv"
)

deletion = normalize_method_names(deletion)
deletion = normalize_dataset_names(deletion)


# -------------------------
# Sparsity
# -------------------------

sparsity = pd.read_csv(SPARSITY_FILE)

check_columns(
    sparsity,
    REQUIRED_COLUMNS["sparsity"],
    "milestone_8_sparsity.csv"
)

sparsity = normalize_method_names(sparsity)
sparsity = normalize_dataset_names(sparsity)


# -------------------------
# Stability
# -------------------------

stability = pd.read_csv(STABILITY_FILE)

check_columns(
    stability,
    REQUIRED_COLUMNS["stability"],
    "milestone_8_stability_summary.csv"
)

stability = normalize_method_names(stability)
stability = normalize_dataset_names(stability)


# -------------------------
# Robustness
# -------------------------

robustness = pd.read_csv(ROBUSTNESS_FILE)

check_columns(
    robustness,
    REQUIRED_COLUMNS["robustness"],
    "milestone_8_robustness.csv"
)

robustness = normalize_method_names(robustness)
robustness = normalize_dataset_names(robustness)


print("✓ All Milestone 8 files loaded successfully.")


# ============================================================
# 1. DELETION FAITHFULNESS SUMMARY
# ============================================================

print("\nCreating deletion faithfulness summary...")

# Keep the complete deletion curve.
deletion_comparison = deletion[
    [
        "dataset",
        "method",
        "removed_features",
        "test_macro_f1",
        "f1_drop",
    ]
].copy()

deletion_comparison = deletion_comparison.sort_values(
    ["dataset", "method", "removed_features"]
)


# ============================================================
# 2. FINAL SPARSITY TABLE
# ============================================================

print("Creating sparsity table...")

sparsity_comparison = sparsity[
    [
        "dataset",
        "method",
        "baseline_macro_f1",
        "threshold_90_percent",
        "sparsity_k",
        "macro_f1_at_sparsity_k",
    ]
].copy()

sparsity_comparison = sparsity_comparison.sort_values(
    ["dataset", "method"]
)


# ============================================================
# 3. STABILITY TABLE
# ============================================================

print("Creating stability table...")

stability_comparison = stability[
    [
        "dataset",
        "method",
        "mean_jaccard",
        "std_jaccard",
    ]
].copy()

stability_comparison = stability_comparison.sort_values(
    ["dataset", "method"]
)


# ============================================================
# 4. ROBUSTNESS TABLE
# ============================================================

print("Creating robustness table...")

robustness_comparison = robustness[
    [
        "dataset",
        "sigma",
        "method",
        "top_k",
        "jaccard_similarity",
    ]
].copy()

robustness_comparison = robustness_comparison.sort_values(
    ["dataset", "sigma", "method"]
)


# ============================================================
# 5. CREATE FINAL 3 × 4 AUDIT GRID
# ============================================================

print("Creating final 3 × 4 audit grid...")


# ------------------------------------------------------------
# Faithfulness
#
# We report the F1 drop at 10 removed features.
# This is one fixed point on the deletion curve and avoids
# creating an arbitrary aggregate score.
# ------------------------------------------------------------

faithfulness_k10 = deletion[
    deletion["removed_features"] == 10
].copy()

faithfulness_k10 = faithfulness_k10[
    [
        "dataset",
        "method",
        "test_macro_f1",
        "f1_drop",
    ]
].rename(
    columns={
        "test_macro_f1": "faithfulness_macro_f1_at_k10",
        "f1_drop": "faithfulness_f1_drop_at_k10",
    }
)


# ------------------------------------------------------------
# Sparsity
# ------------------------------------------------------------

sparsity_grid = sparsity[
    [
        "dataset",
        "method",
        "sparsity_k",
        "macro_f1_at_sparsity_k",
    ]
].copy()


# ------------------------------------------------------------
# Stability
# ------------------------------------------------------------

stability_grid = stability[
    [
        "dataset",
        "method",
        "mean_jaccard",
        "std_jaccard",
    ]
].copy()


# ------------------------------------------------------------
# Robustness
#
# Keep sigma=0.1 and sigma=0.3 separately.
# ------------------------------------------------------------

robustness_pivot = robustness.pivot_table(
    index=["dataset", "method"],
    columns="sigma",
    values="jaccard_similarity",
    aggfunc="mean",
).reset_index()


robustness_pivot.columns.name = None

robustness_pivot = robustness_pivot.rename(
    columns={
        0.1: "robustness_jaccard_sigma_0_1",
        0.3: "robustness_jaccard_sigma_0_3",
    }
)


# ============================================================
# MERGE ALL FOUR DIMENSIONS
# ============================================================

final_grid = faithfulness_k10.merge(
    sparsity_grid,
    on=["dataset", "method"],
    how="outer",
)

final_grid = final_grid.merge(
    stability_grid,
    on=["dataset", "method"],
    how="outer",
)

final_grid = final_grid.merge(
    robustness_pivot,
    on=["dataset", "method"],
    how="outer",
)


# ============================================================
# FINAL COLUMN ORDER
# ============================================================

final_grid = final_grid[
    [
        "dataset",
        "method",

        # Faithfulness
        "faithfulness_macro_f1_at_k10",
        "faithfulness_f1_drop_at_k10",

        # Sparsity
        "sparsity_k",
        "macro_f1_at_sparsity_k",

        # Stability
        "mean_jaccard",
        "std_jaccard",

        # Robustness
        "robustness_jaccard_sigma_0_1",
        "robustness_jaccard_sigma_0_3",
    ]
]


final_grid = final_grid.sort_values(
    ["dataset", "method"]
).reset_index(drop=True)


# ============================================================
# 6. METHOD COMPARISON TABLE
# ============================================================

print("Creating method comparison table...")

method_comparison = final_grid.copy()

method_comparison = method_comparison[
    [
        "dataset",
        "method",
        "faithfulness_f1_drop_at_k10",
        "sparsity_k",
        "mean_jaccard",
        "std_jaccard",
        "robustness_jaccard_sigma_0_1",
        "robustness_jaccard_sigma_0_3",
    ]
]


# ============================================================
# 7. CREATE HUMAN-READABLE SUMMARY
# ============================================================

print("Creating summary table...")

summary_rows = []

for _, row in final_grid.iterrows():

    summary_rows.append(
        {
            "Dataset": row["dataset"],
            "Method": row["method"],

            "Faithfulness: F1 Drop @ k=10":
                row["faithfulness_f1_drop_at_k10"],

            "Sparsity: Minimum k @ 90%":
                row["sparsity_k"],

            "Stability: Mean Jaccard":
                row["mean_jaccard"],

            "Stability: Std Jaccard":
                row["std_jaccard"],

            "Robustness: Jaccard @ sigma=0.1":
                row["robustness_jaccard_sigma_0_1"],

            "Robustness: Jaccard @ sigma=0.3":
                row["robustness_jaccard_sigma_0_3"],
        }
    )


summary = pd.DataFrame(summary_rows)


# ============================================================
# 8. SAVE OUTPUT FILES
# ============================================================

print("\nSaving Milestone 9 outputs...")

final_grid.to_csv(
    RESULTS_DIR / "milestone_9_comparison.csv",
    index=False,
)

summary.to_csv(
    RESULTS_DIR / "milestone_9_summary.csv",
    index=False,
)

method_comparison.to_csv(
    RESULTS_DIR / "milestone_9_method_comparison.csv",
    index=False,
)


# ============================================================
# 9. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL 3 × 4 AUDIT GRID")
print("=" * 70)

print(
    final_grid.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(
    "✓ results/milestone_9_comparison.csv"
)

print(
    "✓ results/milestone_9_summary.csv"
)

print(
    "✓ results/milestone_9_method_comparison.csv"
)


print("\n" + "=" * 70)
print("MILESTONE 9 COMPLETED")
print("=" * 70)