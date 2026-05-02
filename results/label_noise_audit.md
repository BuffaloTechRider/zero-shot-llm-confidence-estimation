# Label Noise Audit

**Primary judge:** us.anthropic.claude-opus-4-5-20251101-v1:0
**Audit judge:** us.anthropic.claude-sonnet-4-5-20250929-v1:0
**Runs audited:** v0.1-qwen-20260427, v0.1-llama-20260428, v0.1-mistral-20260429

| Path | Sampled | Disagreements | Rate |
|---|---:|---:|---:|
| Letter-match | 50 | 9 | 18.0% |
| Judge-fallback | 50 | 6 | 12.0% |

**Pipeline composition:** 17.5% letter-match, 82.5% judge-fallback
**Weighted noise bound:** 13.05%

Interpretation: the weighted noise bound estimates the fraction of
`local_correct` labels that would change if a different judge model
were used. This bounds the label noise that could affect AUROC measurements.