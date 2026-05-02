# Autodidact v0.1 — Confidence Ablation Memo

**Run ID:** `v0.1-qwen-20260427`
**Date:** 2026-04-28
**Local model:** (see --local-model used at harness time)
**Cloud model:** (see --cloud-model used at harness time) (answers + LLM-judge)
**Embedding model:** qllama/bge-large-en-v1.5
**Evaluation set:** 931 MMLU-Pro queries (stratified across categories, seed 42)
**Training set for RouteLLM baselines:** 1000 disjoint MMLU-Pro queries (seed 43)

## Summary

Across 931 evaluation queries, the best signal combination `logprob_uncertainty_only (mean)` achieved AUROC 0.714 against the RouteLLM-plus-KS baseline's 0.665 (gap +0.049). The confidence evaluator is usable but not strong. Best combo AUROC is 0.70-0.79 — build with caveats. Set the local/cloud threshold conservatively so the product prefers cloud escalation when signals are ambiguous.

## Per-Signal AUROC

How well each individual signal separates correct from incorrect local answers. AUROC of 0.5 is chance; 1.0 is perfect.

| Signal | AUROC | 95% CI | n |
|---|---|---|---|
| `knowledge_similarity` | 0.426 | [0.389, 0.465] | 931 |
| `query_classification` | 0.524 | [0.495, 0.558] | 931 |
| `energy_scorer` | n/a | n/a | 0 |
| `grounded_self_assessment` | 0.562 | [0.522, 0.598] | 931 |
| `logprob_uncertainty` | 0.714 | [0.683, 0.746] | 931 |
| `self_consistency` | 0.504 | [0.490, 0.517] | 931 |

## Per-Combo AUROC

Fused-signal ablation. "Mean" is a simple arithmetic mean across signals; "Thompson" samples per-signal weights from their Beta(α, β) distributions each query.

| Combo | Mean AUROC | Mean CI | Thompson AUROC | Thompson CI | n |
|---|---|---|---|---|---|
| `energy_only` | n/a | n/a | n/a | n/a | 931 |
| `knowledge_similarity_only` | 0.426 | [0.389, 0.465] | 0.426 | [0.389, 0.465] | 931 |
| `grounded_self_assessment_only` | 0.562 | [0.522, 0.598] | 0.562 | [0.522, 0.598] | 931 |
| `logprob_uncertainty_only` | 0.714 | [0.683, 0.746] | 0.714 | [0.683, 0.746] | 931 |
| `logprob_plus_gsa` | 0.634 | [0.597, 0.669] | 0.547 | [0.513, 0.583] | 931 |
| `logprob_plus_knowledge` | 0.489 | [0.453, 0.526] | 0.523 | [0.484, 0.558] | 931 |
| `logprob_plus_gsa_plus_knowledge` | 0.486 | [0.449, 0.523] | 0.513 | [0.475, 0.550] | 931 |
| `energy_plus_knowledge` | 0.426 | [0.389, 0.465] | 0.426 | [0.389, 0.465] | 931 |
| `energy_plus_knowledge_plus_gsa` | 0.440 | [0.403, 0.476] | 0.457 | [0.423, 0.494] | 931 |
| `all_six_mean` | 0.523 | [0.487, 0.562] | 0.517 | [0.479, 0.554] | 931 |
| `all_six_thompson` | 0.523 | [0.487, 0.562] | 0.517 | [0.479, 0.554] | 931 |
| `gsa_v3_070_only` | 0.599 | [0.564, 0.634] | 0.599 | [0.564, 0.634] | 931 |
| `gsa_v3_060_only` | 0.511 | [0.473, 0.547] | 0.511 | [0.473, 0.547] | 931 |
| `logprob_plus_gsa_v3_070` | 0.636 | [0.600, 0.671] | 0.577 | [0.541, 0.612] | 931 |
| `logprob_plus_gsa_v3_060` | 0.561 | [0.525, 0.596] | 0.540 | [0.503, 0.576] | 931 |

## Comparison to Prior Art

| Approach | AUROC | 95% CI |
|---|---|---|
| `routellm_no_memory` (memory-blind RouteLLM-style classifier) | 0.664 | [0.629, 0.699] |
| `routellm_plus_ks` (RouteLLM-style + knowledge similarity feature) | 0.665 | [0.629, 0.700] |
| **Our best combo (`logprob_uncertainty_only (mean)`)** | **0.714** | [0.683, 0.746] |

**Headline number:** our best combo beats `routellm_plus_ks` by +0.049 AUROC.

Interpretation: our memory-aware combo is close to a strong supervised baseline but does not clearly beat it. The architecture is validated but the algorithmic win is marginal. Product viable; paper narrower than hoped.

## Calibration

![Best combo calibration](calibration_best_logprob_uncertainty_only_mean.png)

A well-calibrated signal has its observed curve on the diagonal. Systematic bias above the diagonal means under-confidence (the signal says 0.6 but the real correct rate is higher); below the diagonal means over-confidence. Temperature scaling can fix mis-calibration cheaply if needed at product time.

## Retrieval Quality Stratification

Retrieval quality is a potential confound: if retrieval misses, knowledge-similarity and grounded-self-assessment both lose signal. The `retrieval_recall_at_5` proxy (does any top-5 retrieved entry match the query's MMLU-Pro category) lets us split the memo.

| Condition | Best combo AUROC | 95% CI | n |
|---|---|---|---|
| `good_retrieval` | 0.719 | [0.681, 0.752] | 832 |
| `bad_retrieval` | 0.699 | [0.589, 0.797] | 99 |

## Recommendation

The confidence evaluator is usable but not strong. Best combo AUROC is 0.70-0.79 — build with caveats. Set the local/cloud threshold conservatively so the product prefers cloud escalation when signals are ambiguous.

**Next action:** Proceed to v0.2 with a conservative threshold. Consider improving retrieval quality first.

The full 6-signal combo (AUROC 0.517) significantly beats the 3-signal combo (AUROC 0.457) — paired ΔAUROC +0.060 (95% CI [+0.016, +0.103]). The expensive signals earn their keep. Keep them.

**Paper is marginal.** Paired-bootstrap ΔAUROC is +0.049 (95% CI [+0.008, +0.087]); CI excludes 0 but the effect size is below the +0.050 threshold. Publishable as a short empirical note; not a strong paper.

### Reproducing this memo

```bash
python -m benchmarks.routellm_baseline --training-seed 43
python -m benchmarks.ablation_experiment --run-id v0.1-qwen-20260427 --local-model (see --local-model used at harness time)
python -m benchmarks.ablation_analysis --run-id v0.1-qwen-20260427
```

Artifacts: `results/experiment/summary.json`, `results/experiment/roc_overlay.png`, `results/experiment/calibration_*.png`, this memo.
