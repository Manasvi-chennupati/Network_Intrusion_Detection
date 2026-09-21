import pandas as pd
import os

INPUT_FILE = "results/milestone_8_deletion_faithfulness.csv"
OUTPUT_FILE = "results/milestone_8_sparsity.csv"

# 90% threshold
THRESHOLD = 0.90

df = pd.read_csv(INPUT_FILE)

results = []

for (dataset, method), group in df.groupby(["dataset", "method"]):

    group = group.sort_values("removed_features")

    # Full-feature baseline
    baseline = group.loc[
        group["removed_features"] == 0,
        "test_macro_f1"
    ].iloc[0]

    threshold_value = THRESHOLD * baseline

    # Find the smallest k for which remaining F1
    # is still >= 90% of baseline
    valid = group[
        group["test_macro_f1"] >= threshold_value
    ]

    if len(valid) > 0:
        # Largest number of removed features that still
        # retains at least 90% of baseline performance
        row = valid.loc[
            valid["removed_features"].idxmax()
        ]

        k = int(row["removed_features"])
        remaining_f1 = float(row["test_macro_f1"])

    else:
        k = None
        remaining_f1 = None

    results.append({
        "dataset": dataset,
        "method": method,
        "baseline_macro_f1": baseline,
        "90_percent_threshold": threshold_value,
        "max_removed_features_at_90_percent": k,
        "macro_f1_at_k": remaining_f1
    })

results_df = pd.DataFrame(results)

os.makedirs("results", exist_ok=True)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("Milestone 8 — Sparsity")
print("=" * 70)

print(results_df.to_string(index=False))

print(f"\nSaved to: {OUTPUT_FILE}")