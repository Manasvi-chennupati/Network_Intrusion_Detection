# Milestone 10 — Final Audit Results and Discussion

## 10.1 Final Audit Grid

The three explanation methods—SHAP, LIME, and GWO—were evaluated on
NSL-KDD and RT-IoT2022 using four audit dimensions:

1. Deletion faithfulness
2. Sparsity
3. Stability
4. Robustness

The final audit grid is shown below.


| Dataset | Method | F1 Drop @ k=10 | Sparsity k | Mean Jaccard | Robustness σ=0.1 | Robustness σ=0.3 |
|---|---|---:|---:|---:|---:|---:|
| NSL-KDD | GWO | 0.096598 | 3 | 0.052632 | 0.111111 | 0.052632 |
| NSL-KDD | LIME | -0.016787 | 23 | 1.000000 | 0.666667 | 0.000000 |
| NSL-KDD | SHAP | 0.250511 | 13 | 1.000000 | 0.818182 | 0.818182 |
| RT-IoT2022 | GWO | 0.511157 | 33 | 0.054581 | 0.052632 | 0.000000 |
| RT-IoT2022 | LIME | 0.111692 | 75 | 1.000000 | 1.000000 | 1.000000 |
| RT-IoT2022 | SHAP | 0.522669 | 33 | 1.000000 | 0.818182 | 0.818182 |


**Note:** Faithfulness is represented by the change in Macro-F1 after
removing the top 10 ranked features. Sparsity is the minimum number of
ranked features required to recover at least 90% of the baseline
Macro-F1. Stability is the mean bootstrap Jaccard similarity.
Robustness is the Jaccard similarity between the original and
noise-perturbed rankings.

---

## 10.2 Faithfulness

The deletion audit shows that the measured effect of removing highly
ranked features differs across methods and datasets.

For NSL-KDD, removing the top 10 SHAP features produced an F1 drop of
0.250511. The corresponding LIME result produced an F1 drop of
-0.016787, while GWO produced an F1 drop of 0.096598.

For RT-IoT2022, the corresponding F1 drops were 0.522669 for SHAP,
0.111692 for LIME, and 0.511157 for GWO.

Therefore, the faithfulness measurements do not produce one identical
method ordering across the two datasets. The deletion response is
dataset-dependent in the present experiments.

The negative LIME value on NSL-KDD is retained as an observed result.
It indicates that the measured Macro-F1 after the deletion operation
was slightly higher than the baseline at that evaluation point.

---

## 10.3 Sparsity

The sparsity audit measures the minimum number of ranked features
required to recover at least 90% of baseline Macro-F1.

For NSL-KDD, the required feature counts were:

- SHAP: 13
- LIME: 23
- GWO: 3

For RT-IoT2022:

- SHAP: 33
- LIME: 75
- GWO: 33

The smallest observed sparsity value is therefore GWO with k=3 on
NSL-KDD. On RT-IoT2022, GWO and SHAP both require k=33 under the
specified sparsity criterion.

This demonstrates that sparsity is dataset-dependent. The feature
count required to retain 90% of baseline performance changes between
NSL-KDD and RT-IoT2022.

---

## 10.4 Stability

The stability audit shows a repeated pattern across the two datasets.

SHAP and LIME both have a mean Jaccard similarity of 1.0 in the
recorded bootstrap stability audit.

GWO has a substantially lower mean Jaccard similarity:

- NSL-KDD: 0.052632
- RT-IoT2022: 0.054581

Thus, within the implemented bootstrap protocol, high stability is
observed for SHAP and LIME, while substantially lower stability is
observed for GWO on both datasets.

This statement is limited to the evaluated datasets and experimental
configuration.

---

## 10.5 Robustness

The robustness results contain both repeated and dataset-dependent
patterns.

### NSL-KDD

At sigma = 0.1:

- SHAP: 0.818182
- LIME: 0.666667
- GWO: 0.111111

At sigma = 0.3:

- SHAP: 0.818182
- LIME: 0.000000
- GWO: 0.052632

### RT-IoT2022

At sigma = 0.1:

- SHAP: 0.818182
- LIME: 1.000000
- GWO: 0.052632

At sigma = 0.3:

- SHAP: 0.818182
- LIME: 1.000000
- GWO: 0.000000

The SHAP robustness value remains 0.818182 at both tested noise
levels on both datasets.

GWO remains low across the tested noise conditions.

LIME shows different behavior between the two datasets: its
robustness decreases to 0.000000 at sigma = 0.3 on NSL-KDD, while it
remains 1.000000 at both tested noise levels on RT-IoT2022.

Therefore, robustness is partly stable and partly dataset-dependent.

---

## 10.6 Stable Versus Metric-Dependent Findings

### Stable observations

The following patterns are repeated across both datasets:

1. SHAP and LIME have a mean bootstrap Jaccard similarity of 1.0.
2. GWO has a substantially lower mean bootstrap Jaccard similarity on
   both datasets.
3. SHAP has a robustness Jaccard of 0.818182 at both tested noise
   levels on both datasets.
4. GWO robustness remains low across the tested noise conditions.

These are repeated observations within the present experimental
settings and should not be generalized beyond the evaluated datasets
and protocols.

### Metric-dependent observations

The following observations depend on the audit metric and/or dataset:

1. Faithfulness measurements differ between NSL-KDD and RT-IoT2022.
2. The sparsity requirement changes substantially between the two
   datasets.
3. GWO has the smallest sparsity value on NSL-KDD (k=3), whereas GWO
   and SHAP both require k=33 on RT-IoT2022.
4. LIME robustness differs between the two datasets.
5. Performance on one audit dimension does not determine performance
   on another audit dimension.

Therefore, the final grid does not support one single ordering of
SHAP, LIME, and GWO across all four audit dimensions.

---

## 10.7 Overall Interpretation

The experiments demonstrate that evaluating an explanation method
using only one criterion provides an incomplete description of its
behavior.

Faithfulness, sparsity, stability, and robustness measure different
properties of the generated feature rankings. The results show that
these dimensions can exhibit different patterns for the same method.

The two intrusion-detection datasets also produce different
quantitative results. This indicates that the observed explanation
properties depend on both the explanation method and the underlying
dataset/classifier configuration.

The central finding is:

> No single audit metric fully characterizes the quality of a
> feature-importance explanation. The observed behavior of SHAP, LIME,
> and GWO varies across faithfulness, sparsity, stability, and
> robustness, with some patterns remaining consistent across datasets
> and others depending on the evaluation metric and dataset.

Because only three explanation methods were evaluated, the findings
are reported descriptively rather than using directional statistics
to characterize cross-method relationships.

---

## 10.8 Final Research Statement

> This study audits three feature-importance approaches—SHAP, LIME,
> and GWO—using four quantitative criteria on NSL-KDD and RT-IoT2022.
> The results show that explanation behavior is multidimensional:
> faithfulness, sparsity, stability, and robustness do not produce a
> single consistent method ordering. Some patterns are stable across
> datasets, particularly the high bootstrap stability observed for
> SHAP and LIME and the lower bootstrap stability observed for GWO.
> Other results, including deletion faithfulness, sparsity
> requirements, and LIME's robustness, vary with the dataset or
> evaluation metric. These findings support evaluating IDS
> explanations using multiple complementary criteria rather than
> relying on a single feature-importance measure.

---

## 10.9 Reproducibility

Milestone 10 uses the final results generated by Milestone 9 and does
not retrain any models or recompute the XAI methods.

Input:

`results/milestone_9_comparison.csv`

Generated document:

`results/milestone_10_final_writeup.md`

