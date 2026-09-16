# Network_Intrusion_Detection
# Auditing Post-hoc and Metaheuristic Feature Importance for Intrusion Detection

## Milestone 1 — Read & Digest

### Project Overview

This project investigates whether the choice of evaluation protocol changes the perceived quality of different feature-importance methods for intrusion detection.

The experiment uses a fixed XGBoost classifier for each dataset and compares three global feature-importance approaches:

1. **SHAP (TreeSHAP)**
2. **LIME (aggregated LIME ranking)**
3. **Grey Wolf Optimizer (GWO) selection-frequency ranking**

The resulting rankings are evaluated using four quantitative XAI evaluation protocols:

- Faithfulness
- Bootstrap Stability
- Sparsity
- Noise Robustness

The main goal is to determine whether the conclusion about which feature-importance method performs better depends on the evaluation metric being used.

---

### Datasets

The study uses two intrusion-detection datasets:

- **NSL-KDD**
- **RT-IoT2022**

The experimental setup uses the standard train/test split for NSL-KDD.

For RT-IoT2022, an **80/20 stratified train/test split** is created with a fixed random seed.

---

### Feature-Importance Methods

#### 1. SHAP — TreeSHAP

SHAP (SHapley Additive exPlanations) is used to explain the predictions of the fixed XGBoost classifier.

TreeSHAP is used because it is specifically designed for tree-based models and provides feature-attribution values for individual predictions. These values are aggregated to produce a global feature-importance ranking.

#### 2. LIME

LIME (Local Interpretable Model-agnostic Explanations) generates local explanations for individual instances.

A fixed sample of **300 instances** is used, and the local feature contributions are aggregated to produce a global feature-importance ranking.

LIME's run-to-run variability is incorporated into the stability evaluation rather than being ignored.

#### 3. Grey Wolf Optimizer (GWO)

Grey Wolf Optimizer is used as a wrapper-based feature-selection method.

The feature-selection process is performed over **30 independent runs**. The frequency with which each feature is selected is used to construct the GWO global feature-ranking.

Unlike SHAP and LIME, GWO performs a fresh feature-selection search rather than directly explaining the fixed classifier's decision function.

---

### Evaluation Protocols

Each of the three feature rankings is evaluated using four protocols.

| Protocol | Purpose |
|---|---|
| **Faithfulness** | Measures how model performance changes when highly ranked features are removed or masked. |
| **Bootstrap Stability** | Measures the consistency of feature rankings across bootstrap resamples using Jaccard overlap. |
| **Sparsity** | Measures the minimum number of top-ranked features required to recover 90% of macro-F1. |
| **Noise Robustness** | Measures how stable the feature rankings remain when noise is introduced into the input data. |

---

### Controlled Experimental Design

To make the comparison between the three methods consistent, the following are kept fixed:

- Trained XGBoost classifier
- Dataset
- Feature set
- Train/test split
- Evaluation `k` values
- Seed schedule

This allows differences in the evaluation results to be attributed to the feature-importance method and, importantly, to the evaluation protocol rather than to different models or data splits.

---

### Scope Condition

GWO and the two post-hoc explanation methods answer somewhat different questions.

- **SHAP and LIME** explain the behaviour of the fixed XGBoost classifier.
- **GWO** performs a new wrapper-based feature-selection search to identify generally useful features.

This difference is explicitly treated as a **scope condition** of the comparison rather than being hidden or treated as an experimental confound.

---

### Expected Outcomes

The initial expectations are:

- SHAP and LIME may show strong agreement among the top features on NSL-KDD.
- GWO may produce a different top-feature ranking.
- A method's relative position may change depending on the evaluation protocol.
- If all three methods produce the same conclusions across all four protocols and both datasets, this null result would indicate that the choice between SHAP and LIME is less consequential for global feature-importance purposes under these evaluation criteria.

---

### Milestone 1 Status

**Status: Completed — Read & Digest**

The project specification, datasets, feature-importance methods, evaluation protocols, controlled variables, scope condition, and expected outcomes have been reviewed and documented.
