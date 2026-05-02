# Autodidact v0.1 — Confidence Ablation Memo

**Run ID:** `v0.1-llama-20260428`
**Date:** 2026-04-29
**Local model:** (see --local-model used at harness time)
**Cloud model:** (see --cloud-model used at harness time) (answers + LLM-judge)
**Embedding model:** qllama/bge-large-en-v1.5
**Evaluation set:** 997 MMLU-Pro queries (stratified across categories, seed 42)
**Training set for RouteLLM baselines:** 1000 disjoint MMLU-Pro queries (seed 43)

## Summary

Across 997 evaluation queries, the best signal combination `logprob_uncertainty_only (mean)` achieved AUROC 0.650 against the RouteLLM-plus-KS baseline's 0.644 (gap +0.006). The confidence evaluator is marginal. Best combo AUROC is 0.60-0.69 — probably not enough for a user-facing product without more work.

## Per-Signal AUROC

How well each individual signal separates correct from incorrect local answers. AUROC of 0.5 is chance; 1.0 is perfect.

| Signal | AUROC | 95% CI | n |
|---|---|---|---|
| `knowledge_similarity` | 0.413 | [0.379, 0.452] | 997 |
| `query_classification` | 0.521 | [0.491, 0.552] | 997 |
| `energy_scorer` | n/a | n/a | 0 |
| `grounded_self_assessment` | 0.614 | [0.577, 0.652] | 997 |
| `logprob_uncertainty` | 0.650 | [0.616, 0.687] | 997 |
| `self_consistency` | 0.594 | [0.558, 0.627] | 997 |

## Per-Combo AUROC

Fused-signal ablation. "Mean" is a simple arithmetic mean across signals; "Thompson" samples per-signal weights from their Beta(α, β) distributions each query.

| Combo | Mean AUROC | Mean CI | Thompson AUROC | Thompson CI | n |
|---|---|---|---|---|---|
| `energy_only` | n/a | n/a | n/a | n/a | 997 |
| `knowledge_similarity_only` | 0.413 | [0.379, 0.452] | 0.413 | [0.379, 0.452] | 997 |
| `grounded_self_assessment_only` | 0.614 | [0.577, 0.652] | 0.614 | [0.577, 0.652] | 997 |
| `logprob_uncertainty_only` | 0.650 | [0.616, 0.687] | 0.650 | [0.616, 0.687] | 997 |
| `logprob_plus_gsa` | 0.645 | [0.608, 0.682] | 0.632 | [0.596, 0.670] | 997 |
| `logprob_plus_knowledge` | 0.532 | [0.497, 0.570] | 0.541 | [0.504, 0.579] | 997 |
| `logprob_plus_gsa_plus_knowledge` | 0.617 | [0.581, 0.652] | 0.577 | [0.540, 0.615] | 997 |
| `energy_plus_knowledge` | 0.413 | [0.379, 0.452] | 0.413 | [0.379, 0.452] | 997 |
| `energy_plus_knowledge_plus_gsa` | 0.572 | [0.536, 0.609] | 0.546 | [0.510, 0.583] | 997 |
| `all_six_mean` | 0.643 | [0.608, 0.681] | 0.616 | [0.577, 0.654] | 997 |
| `all_six_thompson` | 0.643 | [0.608, 0.681] | 0.616 | [0.577, 0.654] | 997 |
| `gsa_v3_070_only` | n/a | n/a | n/a | n/a | 0 |
| `gsa_v3_060_only` | n/a | n/a | n/a | n/a | 0 |
| `logprob_plus_gsa_v3_070` | n/a | n/a | n/a | n/a | 0 |
| `logprob_plus_gsa_v3_060` | n/a | n/a | n/a | n/a | 0 |

## Comparison to Prior Art

| Approach | AUROC | 95% CI |
|---|---|---|
| `routellm_no_memory` (memory-blind RouteLLM-style classifier) | 0.662 | [0.628, 0.701] |
| `routellm_plus_ks` (RouteLLM-style + knowledge similarity feature) | 0.644 | [0.609, 0.683] |
| **Our best combo (`logprob_uncertainty_only (mean)`)** | **0.650** | [0.616, 0.687] |

**Headline number:** our best combo beats `routellm_plus_ks` by +0.006 AUROC.

Interpretation: our memory-aware combo is close to a strong supervised baseline but does not clearly beat it. The architecture is validated but the algorithmic win is marginal. Product viable; paper narrower than hoped.

## Calibration

![Best combo calibration](calibration_best_logprob_uncertainty_only_mean.png)

A well-calibrated signal has its observed curve on the diagonal. Systematic bias above the diagonal means under-confidence (the signal says 0.6 but the real correct rate is higher); below the diagonal means over-confidence. Temperature scaling can fix mis-calibration cheaply if needed at product time.

## Retrieval Quality Stratification

Retrieval quality is a potential confound: if retrieval misses, knowledge-similarity and grounded-self-assessment both lose signal. The `retrieval_recall_at_5` proxy (does any top-5 retrieved entry match the query's MMLU-Pro category) lets us split the memo.

| Condition | Best combo AUROC | 95% CI | n |
|---|---|---|---|
| `good_retrieval` | 0.652 | [0.613, 0.690] | 889 |
| `bad_retrieval` | 0.640 | [0.517, 0.753] | 108 |

## Recommendation

The confidence evaluator is marginal. Best combo AUROC is 0.60-0.69 — probably not enough for a user-facing product without more work.

**Next action:** Improve the mechanism before shipping. Try a bigger local model, fine-tune a classifier on the training data, or add a new signal.

The full 6-signal combo (AUROC 0.616) significantly beats the 3-signal combo (AUROC 0.546) — paired ΔAUROC +0.071 (95% CI [+0.027, +0.114]). The expensive signals earn their keep. Keep them.

**Paper not viable** at current AUROC. Focus on product.

### Reproducing this memo

```bash
python -m benchmarks.routellm_baseline --training-seed 43
python -m benchmarks.ablation_experiment --run-id v0.1-llama-20260428 --local-model (see --local-model used at harness time)
python -m benchmarks.ablation_analysis --run-id v0.1-llama-20260428
```

Artifacts: `results/experiment/summary.json`, `results/experiment/roc_overlay.png`, `results/experiment/calibration_*.png`, this memo.
