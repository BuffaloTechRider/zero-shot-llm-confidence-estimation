# Autodidact v0.1 — Confidence Ablation Memo

**Run ID:** `v0.1-mistral-20260429`
**Date:** 2026-04-30
**Local model:** (see --local-model used at harness time)
**Cloud model:** (see --cloud-model used at harness time) (answers + LLM-judge)
**Embedding model:** qllama/bge-large-en-v1.5
**Evaluation set:** 999 MMLU-Pro queries (stratified across categories, seed 42)
**Training set for RouteLLM baselines:** 1000 disjoint MMLU-Pro queries (seed 43)

## Summary

Across 999 evaluation queries, the best signal combination `logprob_uncertainty_only (mean)` achieved AUROC 0.678 against the RouteLLM-plus-KS baseline's 0.676 (gap +0.002). The confidence evaluator is marginal. Best combo AUROC is 0.60-0.69 — probably not enough for a user-facing product without more work.

## Per-Signal AUROC

How well each individual signal separates correct from incorrect local answers. AUROC of 0.5 is chance; 1.0 is perfect.

| Signal | AUROC | 95% CI | n |
|---|---|---|---|
| `knowledge_similarity` | 0.422 | [0.385, 0.458] | 999 |
| `query_classification` | 0.522 | [0.489, 0.553] | 999 |
| `energy_scorer` | n/a | n/a | 0 |
| `grounded_self_assessment` | 0.638 | [0.601, 0.673] | 999 |
| `logprob_uncertainty` | 0.678 | [0.644, 0.716] | 999 |
| `self_consistency` | 0.604 | [0.572, 0.635] | 999 |

## Per-Combo AUROC

Fused-signal ablation. "Mean" is a simple arithmetic mean across signals; "Thompson" samples per-signal weights from their Beta(α, β) distributions each query.

| Combo | Mean AUROC | Mean CI | Thompson AUROC | Thompson CI | n |
|---|---|---|---|---|---|
| `energy_only` | n/a | n/a | n/a | n/a | 999 |
| `knowledge_similarity_only` | 0.422 | [0.385, 0.458] | 0.422 | [0.385, 0.458] | 999 |
| `grounded_self_assessment_only` | 0.638 | [0.601, 0.673] | 0.638 | [0.601, 0.673] | 999 |
| `logprob_uncertainty_only` | 0.678 | [0.644, 0.716] | 0.678 | [0.644, 0.716] | 999 |
| `logprob_plus_gsa` | 0.669 | [0.631, 0.705] | 0.580 | [0.544, 0.617] | 999 |
| `logprob_plus_knowledge` | 0.446 | [0.407, 0.482] | 0.507 | [0.467, 0.545] | 999 |
| `logprob_plus_gsa_plus_knowledge` | 0.524 | [0.487, 0.563] | 0.541 | [0.502, 0.581] | 999 |
| `energy_plus_knowledge` | 0.422 | [0.385, 0.458] | 0.422 | [0.385, 0.458] | 999 |
| `energy_plus_knowledge_plus_gsa` | 0.505 | [0.468, 0.545] | 0.499 | [0.464, 0.538] | 999 |
| `all_six_mean` | 0.621 | [0.585, 0.656] | 0.599 | [0.562, 0.634] | 999 |
| `all_six_thompson` | 0.621 | [0.585, 0.656] | 0.599 | [0.562, 0.634] | 999 |
| `gsa_v3_070_only` | n/a | n/a | n/a | n/a | 0 |
| `gsa_v3_060_only` | n/a | n/a | n/a | n/a | 0 |
| `logprob_plus_gsa_v3_070` | n/a | n/a | n/a | n/a | 0 |
| `logprob_plus_gsa_v3_060` | n/a | n/a | n/a | n/a | 0 |

## Comparison to Prior Art

| Approach | AUROC | 95% CI |
|---|---|---|
| `routellm_no_memory` (memory-blind RouteLLM-style classifier) | 0.676 | [0.640, 0.710] |
| `routellm_plus_ks` (RouteLLM-style + knowledge similarity feature) | 0.676 | [0.640, 0.710] |
| **Our best combo (`logprob_uncertainty_only (mean)`)** | **0.678** | [0.644, 0.716] |

**Headline number:** our best combo beats `routellm_plus_ks` by +0.002 AUROC.

Interpretation: our memory-aware combo is close to a strong supervised baseline but does not clearly beat it. The architecture is validated but the algorithmic win is marginal. Product viable; paper narrower than hoped.

## Calibration

![Best combo calibration](calibration_best_logprob_uncertainty_only_mean.png)

A well-calibrated signal has its observed curve on the diagonal. Systematic bias above the diagonal means under-confidence (the signal says 0.6 but the real correct rate is higher); below the diagonal means over-confidence. Temperature scaling can fix mis-calibration cheaply if needed at product time.

## Retrieval Quality Stratification

Retrieval quality is a potential confound: if retrieval misses, knowledge-similarity and grounded-self-assessment both lose signal. The `retrieval_recall_at_5` proxy (does any top-5 retrieved entry match the query's MMLU-Pro category) lets us split the memo.

| Condition | Best combo AUROC | 95% CI | n |
|---|---|---|---|
| `good_retrieval` | 0.684 | [0.645, 0.721] | 891 |
| `bad_retrieval` | 0.627 | [0.493, 0.764] | 108 |

## Recommendation

The confidence evaluator is marginal. Best combo AUROC is 0.60-0.69 — probably not enough for a user-facing product without more work.

**Next action:** Improve the mechanism before shipping. Try a bigger local model, fine-tune a classifier on the training data, or add a new signal.

The full 6-signal combo (AUROC 0.599) significantly beats the 3-signal combo (AUROC 0.499) — paired ΔAUROC +0.100 (95% CI [+0.054, +0.145]). The expensive signals earn their keep. Keep them.

**Paper not viable** at current AUROC. Focus on product.

### Reproducing this memo

```bash
python -m benchmarks.routellm_baseline --training-seed 43
python -m benchmarks.ablation_experiment --run-id v0.1-mistral-20260429 --local-model (see --local-model used at harness time)
python -m benchmarks.ablation_analysis --run-id v0.1-mistral-20260429
```

Artifacts: `results/experiment/summary.json`, `results/experiment/roc_overlay.png`, `results/experiment/calibration_*.png`, this memo.
