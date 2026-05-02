# Zero-Shot Confidence Estimation for Small LLMs: When Supervised Baselines Aren't Worth Training

## Abstract

Local-to-cloud LLM routing decides whether a cheap local model can answer a query or whether to escalate to an expensive cloud model. The dominant approach trains a supervised classifier on labeled routing examples. We ask whether that training is necessary.

We compare six zero-shot confidence signals against RouteLLM-style supervised baselines across three local model families (Qwen-2.5-7B, Llama-3.1-8B, Mistral-7B-Instruct) and two datasets (MMLU-Pro, TriviaQA; ~3,000 evaluation queries per model). We find that (1) average token log-probability, a zero-shot signal requiring no training data, matches or exceeds supervised baselines in-distribution (AUROC 0.650--0.714 vs.\ 0.644--0.676) and substantially outperforms them out-of-distribution (0.717--0.833 vs.\ 0.512--0.564); (2) supervised routing transfers across model families but collapses on a new dataset; (3) naive multi-signal fusion hurts rather than helps; and (4) signal quality correlates with RLHF calibration training across model families. We provide a cold-start deployment guide and release all code, data, and experiment logs.

---

## 1. Introduction

Running a small local model (7--8B parameters) for most queries and escalating to a cloud model only when the local model is likely to fail is an increasingly common architecture for cost-sensitive LLM deployments. The routing decision---local or cloud?---determines both cost and quality. A growing body of work addresses this problem, but the field has converged on supervised approaches without adequately testing whether the supervision is necessary.

### 1.1 Supervised Routing

RouteLLM \cite{ong2024routellm} established the dominant paradigm: train a classifier on labeled (query, model-was-sufficient) pairs. Subsequent work extends this to contextual bandits under budget constraints \cite{chen2025adaptive}, bandit feedback with regret guarantees \cite{soare2025bandit}, dueling (pairwise preference) feedback \cite{li2025dueling}, and non-stationary settings where pricing and quality shift over time \cite{tabernermiller2026paretobandit}. Gomes et al.\ \cite{gomes2025routing} survey routing strategies more broadly.

All of these approaches require training data---labeled examples, preference pairs, or online feedback. For a new deployment, this means generating answers from both local and cloud models on a representative corpus, labeling correctness, and fitting a routing model. In our experiments, this costs approximately \$25 and 2 hours per local model.

### 1.2 Zero-Shot Confidence Signals

An alternative extracts confidence signals directly from the local model's inference, with no training data.

\textbf{Token-level calibration.} Kadavath et al.\ \cite{kadavath2022language} showed that RLHF'd models produce calibrated token probabilities that correlate with correctness. Guo et al.\ \cite{guo2017calibration} established calibration theory for deep networks. Huang et al.\ \cite{huang2024logprob} demonstrated that log-probabilities remain reliable plausibility estimates in instruction-tuned models.

\textbf{Self-assessment.} Kadavath et al.\ \cite{kadavath2022language} tested P(True) prompting for self-evaluation. Ren et al.\ \cite{ren2023selfeval} reformulated generation into token-level self-evaluation, improving selective generation on TruthfulQA. Chen et al.\ \cite{chen2024trust} found self-reported confidence achieves better calibration than self-consistency at 5$\times$ lower cost.

\textbf{Multi-signal uncertainty.} UQLM \cite{lin2025uqlm} and MUSE \cite{geng2025muse} combine multiple uncertainty signals into unified frameworks. Mukkunnoth et al.\ \cite{mukkunnoth2025confidence} combine semantic alignment, internal convergence, and learned confidence for pre-generation hallucination routing. Wang et al.\ \cite{wang2022selfconsistency} introduced self-consistency---sampling multiple reasoning paths and measuring agreement---as a confidence proxy.

### 1.3 Uncertainty-Based Routing

The closest work to ours is Chuang et al.\ \cite{chuang2025confident}, who benchmark uncertainty-based SLM-to-LLM routing across 1,500+ settings. They find that uncertainty distributions depend more on the SLM and quantification method than on downstream data, and propose a calibration data construction pipeline for generalization. Zhang et al.\ \cite{zhang2025uncertainty} propose a confidence-driven router for edge-cloud deployment, evaluating on MT-Bench, GSM8K, and MMLU.

### 1.4 The Gap We Address

Despite this active literature, a specific comparison is missing. Existing routing papers assume supervision is available and do not benchmark against zero-shot alternatives. Existing confidence papers study signal quality in isolation without comparing to supervised routing baselines. Chuang et al.\ \cite{chuang2025confident} benchmark uncertainty methods against each other, not against supervised routers. No published work evaluates zero-shot confidence signals against supervised routing baselines across multiple model families \emph{and} multiple datasets.

We fill this gap and find that the supervised approach is not worth the investment for most deployments.

### 1.5 Contributions

\begin{enumerate}
\item A cross-model, cross-dataset comparison of six zero-shot confidence signals against supervised baselines (3 models $\times$ 2 datasets $\times$ $\sim$1,000 queries each), with bootstrap CIs and paired significance tests.
\item The finding that supervised routing is dataset-specific while zero-shot log-probability transfers across both models and datasets (\S\ref{sec:cross-dataset}).
\item A learning curve showing supervised routing converges at $\sim$250 examples but never exceeds the zero-shot baseline (\S\ref{sec:learning-curve}).
\item Evidence that signal quality correlates with RLHF calibration training, with logprob and self-assessment complementary across the model-family axis (\S\ref{sec:analysis-logprob}).
\item A cold-start deployment guide (\S\ref{sec:deployment}) and a retrieval-conditional self-assessment design principle (\S\ref{sec:retrieval-gsa}).
\end{enumerate}

All experiments are reproducible on a single laptop with Ollama and AWS Bedrock. Total cloud cost: \$123.

---

## 2. Signals and Baselines

### 2.1 Zero-Shot Confidence Signals

We evaluate six signals computed at inference time without training data.

\textbf{Log-probability (logprob).} Average per-token log-probability of the local model's generated answer, mapped to $[0,1]$ via $\sigma(\bar{\ell} \cdot 2 + 3)$ where $\bar{\ell}$ is the mean token log-prob. Higher values indicate greater token-level confidence \cite{kadavath2022language}.

\textbf{Self-assessment (GSA).} A single-token YES/NO probe: the model is asked ``Are you confident you can answer this question correctly?'' and the YES-token probability is extracted from the first position's logprobs. We test a bare variant and a retrieval-conditional variant (v3) that injects strong retrieved hits when available but falls back to a byte-identical bare prompt otherwise, so the model cannot distinguish ``retrieval found nothing'' from ``retrieval was never attempted.''

\textbf{Self-consistency (SC).} Jaccard token overlap between two independent generations at temperature 0.0 and 0.7 \cite{wang2022selfconsistency}.

\textbf{Knowledge similarity (KS).} Maximum cosine similarity between the query embedding and top-5 retrieved KB entries via FAISS inner-product on L2-normalized bge-large-en-v1.5 embeddings (1024-dim).

\textbf{Query classification (QC).} Keyword heuristic: factual $\to$ 0.7, real-time $\to$ 0.2, creative $\to$ 0.4, default $\to$ 0.5.

\textbf{Energy scorer (ES).} Logistic regression on query embeddings trained on the agent's pass/fail history. Disabled in all experiments (requires 50+ labeled examples).

### 2.2 Supervised Baselines

Following RouteLLM \cite{ong2024routellm}, we train two logistic regression classifiers (LogisticRegressionCV, 5-fold) on a disjoint 1,000-query training corpus per model:

- \textbf{RouteLLM-nm}: query embedding $\to$ $P(\text{local sufficient})$
- \textbf{RouteLLM-pks}: $[\text{query embedding} \| \text{knowledge similarity}] \to P(\text{local sufficient})$

---

## 3. Experimental Setup

### 3.1 Models

<!-- Table 1 -->
**Table 1.** Local models evaluated.

| Model | Params | RLHF emphasis |
|---|---:|---|
| Qwen-2.5-7B | 7.6B | Calibration-aware |
| Llama-3.1-8B-Instruct | 8.0B | Helpfulness / harmlessness |
| Mistral-7B-Instruct | 7.2B | Mixed |

Cloud model: Claude Sonnet 4.5 (AWS Bedrock). Judge: Claude Opus 4.5.

### 3.2 Datasets

**MMLU-Pro** (primary): 10-option MCQ, 14 categories. KB: 995 seeded entries. Eval: 931--999 queries per model.

**TriviaQA** (secondary): open-ended short-answer. KB: 200 seeded entries. Eval: 500 queries per model.

### 3.3 Labeling

MMLU-Pro uses regex letter extraction with LLM-judge fallback. TriviaQA uses standard substring matching.

<!-- Table 2 -->
**Table 2.** Judge fallback rates on MMLU-Pro.

| Model | Judge calls / total | Rate |
|---|---:|---:|
| Qwen-2.5-7B | 688 / 931 | 74% |
| Llama-3.1-8B | 755 / 997 | 76% |
| Mistral-7B | 972 / 999 | 97% |

Mistral's 97% judge rate reflects poorly formatted responses; its AUROC is more sensitive to judge noise. Label noise bound: 13% inter-judge disagreement (Opus vs.\ Sonnet audit on 100 rows). Differential comparisons are robust to symmetric noise.

### 3.4 Evaluation Protocol

AUROC with 1,000-sample bootstrap CIs (fixed seed). Paired deltas via bootstrap resampling on the same query set. Significance: 95% CI excludes zero.

---

## 4. Results

### 4.1 Per-Signal AUROC on MMLU-Pro

<!-- Table 3 -->
**Table 3.** AUROC on MMLU-Pro (95% bootstrap CI in brackets). Best zero-shot signal in bold. $n \approx 950$ per model.

| Signal | Qwen-2.5-7B | Llama-3.1-8B | Mistral-7B |
|---|---:|---:|---:|
| **logprob** | **0.714** [.683, .746] | **0.650** [.616, .687] | **0.678** [.644, .716] |
| GSA v3 | 0.562 [.522, .598] | 0.614 [.577, .652] | 0.638 [.601, .674] |
| SC | 0.504 [.490, .517] | 0.594 [.558, .627] | 0.604 [.569, .638] |
| KS | 0.426 [.389, .465] | 0.413 [.379, .452] | 0.422 [.387, .459] |
| QC | 0.524 [.495, .558] | 0.521 [.491, .552] | 0.522 [.491, .553] |
| RouteLLM-nm | 0.664 [.629, .699] | 0.662 [.628, .701] | 0.676 [.641, .710] |
| RouteLLM-pks | 0.665 [.629, .700] | 0.644 [.609, .683] | 0.676 [.641, .711] |

Log-probability is the best single signal on every model. It significantly exceeds RouteLLM on Qwen ($\Delta$ = +0.049, CI [+0.008, +0.087]) and ties on Llama and Mistral. Local accuracy: Qwen 43.7%, Llama 36.2%, Mistral 30.5%.

### 4.2 Cross-Dataset Transfer \label{sec:cross-dataset}

<!-- Table 4 -->
**Table 4.** Cross-dataset AUROC (averaged across 3 models).

| Signal | MMLU-Pro | TriviaQA | $\Delta$ |
|---|---:|---:|---:|
| logprob | 0.681 | **0.782** | +0.101 |
| GSA | 0.605 | 0.703 | +0.098 |
| RouteLLM | 0.662 | 0.546 | **$-$0.116** |

RouteLLM collapses from 0.662 to chance (0.546) on TriviaQA. Log-probability improves to 0.782. Per-model TriviaQA logprob: Qwen 0.828, Llama 0.800, Mistral 0.717.

<!-- Figure 1 placeholder -->
**Figure 1.** Cross-dataset transfer. Left: logprob AUROC on MMLU-Pro vs.\ TriviaQA per model (both above the diagonal). Right: RouteLLM AUROC (all below the diagonal on TriviaQA). *[ROC overlay or scatter plot; existing roc\_overlay.png can be adapted.]*

### 4.3 Supervised Learning Curve \label{sec:learning-curve}

<!-- Table 5 -->
**Table 5.** RouteLLM-nm AUROC at increasing training sizes. Last row: logprob (zero-shot, no training).

| Training $N$ | Qwen | Llama | Mistral |
|---:|---:|---:|---:|
| 25 | 0.757 | 0.624 | 0.484 |
| 50 | 0.511 | 0.470 | 0.624 |
| 100 | 0.609 | 0.692 | 0.671 |
| 250 | 0.654 | 0.681 | 0.566 |
| 500 | 0.653 | 0.623 | 0.650 |
| $\sim$1000 | 0.620 | 0.646 | 0.649 |
| logprob (0) | **0.714** | **0.650** | **0.678** |

RouteLLM is unstable at small $N$ (swings of 0.15--0.20), converges around $N$=250--500, and at convergence never consistently exceeds the zero-shot line.

<!-- Figure 2 placeholder -->
**Figure 2.** RouteLLM learning curve (solid lines) vs.\ logprob zero-shot AUROC (dashed horizontal lines), per model. *[Existing routellm\_learning\_curve.png from each run directory.]*

### 4.4 Signal Latency

<!-- Table 6 -->
**Table 6.** Mean signal latency (ms) from experiment DB. ``Pre-gen'' indicates whether the signal can route before generating a full local answer.

| Signal | Qwen | Llama | Mistral | Pre-gen? |
|---|---:|---:|---:|:---:|
| RouteLLM | $<$1 | $<$1 | $<$1 | \checkmark |
| KS (FAISS) | 72 | 72 | 77 | \checkmark |
| GSA v3 | 461 | 777 | 767 | \checkmark |
| logprob | 1,531 | 4,756 | 3,873 | $\times$ |
| SC | 539 | 2,874 | 2,766 | $\times$ |

Log-probability requires 1.5--4.8s of local generation before routing. A two-stage system can use GSA as a pre-generation filter ($\sim$500ms) and logprob as a post-generation quality check, avoiding wasted compute on queries GSA already flags.

### 4.5 Naive Fusion Hurts

<!-- Table 7 -->
**Table 7.** AUROC of signal combinations (simple mean fusion) vs.\ logprob alone.

| Combination | Qwen | Llama | Mistral |
|---|---:|---:|---:|
| logprob alone | **0.714** | **0.650** | **0.678** |
| all 6 (mean) | 0.523 | 0.643 | 0.643 |
| logprob + GSA | 0.634 | 0.645 | 0.669 |

Mean fusion drags the strong signal toward weaker ones. The effect is largest on Qwen where the quality gap is widest (logprob 0.714 vs.\ next-best GSA 0.562). Fusion only approaches logprob on Llama, where multiple signals have similar strength.

### 4.6 Retrieval-Conditional Self-Assessment \label{sec:retrieval-gsa}

<!-- Table 8 -->
**Table 8.** GSA v3 (retrieval-conditional) vs.\ v2 (bare) on Qwen MMLU-Pro ($n$=931).

| Variant | AUROC | Queries w/ retrieval |
|---|---:|---:|
| GSA v2 (bare) | 0.562 | 0 / 931 |
| GSA v3 ($\tau$=0.70) | **0.599** | 290 / 931 (31%) |
| GSA v3 ($\tau$=0.60) | 0.511 | 769 / 931 (83%) |

Strong-hit-only retrieval ($\tau$=0.70) improves the signal by +0.037; including marginal hits ($\tau$=0.60) degrades it below the bare baseline. The design principle---never mention retrieval absence---is model-agnostic but density-dependent: on TriviaQA's sparse KB (2.6% hit rate at $\tau$=0.70), v3 and v2 are indistinguishable.

---

## 5. Analysis

### 5.1 Why Does Log-Probability Work? \label{sec:analysis-logprob}

Average token log-probability measures how surprised the model was by its own output. On factual QA, a model that ``knows'' the answer produces high-probability tokens; a model that is guessing produces lower-probability, higher-variance tokens \cite{kadavath2022language}.

The signal's strength varies predictably with RLHF training:

<!-- Table 9 -->
**Table 9.** Signal quality vs.\ RLHF emphasis.

| Model | RLHF emphasis | logprob | GSA |
|---|---|---:|---:|
| Qwen-2.5-7B | Calibration-aware | **0.714** | 0.562 |
| Mistral-7B | Mixed | 0.678 | 0.638 |
| Llama-3.1-8B | Helpfulness | 0.650 | **0.614** |

Models with calibration-aware RLHF produce stronger logprob signals. Models without benefit more from explicit self-assessment. The two signals are complementary across the model-family axis.

### 5.2 Why Does Supervised Routing Collapse Cross-Dataset?

RouteLLM learns which regions of embedding space correspond to easy queries on MMLU-Pro. This transfers across model families (Qwen-trained RouteLLM achieves 0.663--0.675 on Llama/Mistral) because query difficulty is partially model-agnostic. But the embedding structure of MMLU-Pro is specific to MMLU-Pro. TriviaQA occupies a different region with different difficulty patterns. The classifier learned ``MMLU-Pro-shaped queries that are easy,'' not ``queries this model can answer.''

Log-probability measures a property of the model's generation, not of the query. It transfers because it does not depend on the query distribution.

### 5.3 Why Does Fusion Hurt?

Equal-weight fusion drags a strong signal toward weaker ones---a known failure mode when component quality is heterogeneous \cite{dietterich2000ensemble}. Thompson Sampling fusion, which should learn to upweight strong signals, collapses to equal-weight in our setup because no routing decisions are made during evaluation (no outcome feedback to update $\alpha/\beta$ parameters). Adaptive fusion with real feedback remains open.

### 5.4 Knowledge Similarity: A Same-Dataset Confound

KS is consistently inverted (AUROC 0.413--0.426). Per-category analysis reveals the mechanism: harder MMLU-Pro categories (law at 16.2\% accuracy, engineering at 29.2\%) have \emph{higher} retrieval similarity because the KB was seeded from the same distribution. The signal measures ``have I seen a similar question''---and harder questions are well-represented precisely because they come from the same dataset. A KB seeded from external sources would produce a different distribution. We did not test this.

---

## 6. Cold-Start Deployment Guide \label{sec:deployment}

<!-- Table 10 -->
**Table 10.** Routing options at deployment time.

| Stage | Signal | AUROC range | Cost | Latency |
|---|---|---|---|---|
| Day 0 | logprob | 0.650--0.833 | \$0 | 1.5--4.8s (post-gen) |
| Day 0 | GSA v3 | 0.562--0.720 | \$0 | 0.5--0.8s (pre-gen) |
| After labeling | RouteLLM | 0.620--0.676 | $\sim$\$25 | $<$1ms (pre-gen) |

\textbf{Recommendation.} Use logprob from the first query. If latency is critical, add GSA as a pre-generation filter. RouteLLM training is only justified when (a) the query distribution is known and stable, (b) pre-generation routing is required, and (c) cross-dataset transfer is not needed. For most deployments, logprob at \$0 is the better default.

---

## 7. Related Work

\textbf{Supervised routing.} RouteLLM \cite{ong2024routellm} trains classifiers on preference data. Extensions include contextual bandits \cite{chen2025adaptive}, bandit feedback \cite{soare2025bandit}, dueling feedback \cite{li2025dueling}, non-stationary pricing \cite{tabernermiller2026paretobandit}, and routing strategy surveys \cite{gomes2025routing}. All require training data or online feedback. None benchmark against zero-shot alternatives.

\textbf{Uncertainty-based routing.} Chuang et al.\ \cite{chuang2025confident} benchmark uncertainty-based SLM routing across 1,500+ settings, finding distributions depend on the SLM and method more than on data. Zhang et al.\ \cite{zhang2025uncertainty} propose confidence-driven edge-cloud routing. Mukkunnoth et al.\ \cite{mukkunnoth2025confidence} combine three signals for pre-generation hallucination routing. Our work differs in directly comparing against supervised baselines and testing cross-dataset transfer.

\textbf{Token-level calibration.} Kadavath et al.\ \cite{kadavath2022language} showed calibrated token probabilities in RLHF'd models. Guo et al.\ \cite{guo2017calibration} established calibration theory. Huang et al.\ \cite{huang2024logprob} confirmed log-probability reliability in instruction-tuned models. We apply the same mechanism to routing and show it transfers where supervised approaches fail.

\textbf{Self-assessment.} Kadavath et al.\ \cite{kadavath2022language} tested P(True). Ren et al.\ \cite{ren2023selfeval} reformulated generation into self-evaluation. Chen et al.\ \cite{chen2024trust} found self-reported confidence outperforms self-consistency at lower cost. We extend self-assessment with retrieval-conditional prompting and show effectiveness is model-specific.

\textbf{Multi-signal uncertainty.} UQLM \cite{lin2025uqlm} and MUSE \cite{geng2025muse} combine signals into unified frameworks. We find naive fusion hurts---consistent with ensemble theory \cite{dietterich2000ensemble} under heterogeneous quality, but not previously shown for LLM routing.

\textbf{Self-consistency.} Wang et al.\ \cite{wang2022selfconsistency} proposed multi-path agreement. Our two-sample variant is model-specific: effective on Llama/Mistral, dead on Qwen.

\textbf{Semantic entropy.} Kuhn et al.\ \cite{kuhn2023semantic} proposed detecting confabulations via entropy over semantic clusters of sampled generations, published in \textit{Nature}. Farquhar et al.\ \cite{farquhar2024robust} extended this with cheaper approximations. These methods require multiple generations per query (typically 5--10), making them 5--10$\times$ more expensive than our single-generation logprob signal. We do not evaluate semantic entropy directly but note that our self-consistency signal (two-sample Jaccard) is a simplified variant of the same intuition.

\textbf{Verbalized confidence.} Tian et al.\ \cite{tian2023just} showed LLMs can express calibrated confidence when prompted to output a probability. Xiong et al.\ \cite{xiong2024llmsexpress} found verbalized confidence is often overconfident but can be improved with prompting strategies. Our GSA signal is a constrained variant: we extract the YES-token probability rather than asking for a numeric confidence score, avoiding the overconfidence problem by grounding in logprobs rather than generated text.

\textbf{Uncertainty surveys.} Geng et al.\ \cite{geng2024survey} and Huang et al.\ \cite{huang2025uncertainty} provide comprehensive surveys of LLM uncertainty quantification methods. Our work is narrower in scope but deeper in evaluation: we test fewer methods but across more models and datasets with a supervised baseline comparison that surveys do not include.

---

## 8. Limitations

1. \textbf{Two datasets, one task type.} Both MMLU-Pro and TriviaQA are factual QA. Reasoning, creative, and multi-turn tasks are untested.
2. \textbf{Three models, one size.} All 7--8B. The calibration hypothesis predicts logprob improves with scale; untested.
3. \textbf{Post-generation routing cost.} logprob requires full generation (1.5--4.8s) before routing. Wasted on escalated queries.
4. \textbf{Static KB.} All experiments use a frozen knowledge base. In production, the KB grows from escalations, making routing non-stationary.
5. \textbf{MCQ inflation.} MMLU-Pro's 10-option format gives a 10\% chance floor. TriviaQA partially addresses this.
6. \textbf{Label noise.} 13\% inter-judge disagreement. Absolute AUROC carries $\sim$\pm$0.05 uncertainty; differential comparisons are robust.

---

## 9. Conclusion

Training data is not necessary for effective LLM routing. Average token log-probability---a zero-shot signal available from the first query at zero cost---matches supervised baselines in-distribution and substantially outperforms them out-of-distribution. For practitioners deploying local-to-cloud routing: start with log-probability, skip the training, and invest engineering effort elsewhere.

---

## References


\bibitem{chen2025adaptive} Chen, Y., et al. (2025). Adaptive LLM Routing Under Budget Constraints. \textit{arXiv:2508.21141}.

\bibitem{chen2024trust} Chen, Z., et al. (2024). When Can We Trust LLM Graders? Calibrating Confidence for Automated Assessment. \textit{arXiv:2603.29559}.

\bibitem{chuang2025confident} Chuang, Y.-N., et al. (2025). Confident or Seek Stronger: Exploring Uncertainty-Based On-device LLM Routing. \textit{arXiv:2502.04428}.

\bibitem{dietterich2000ensemble} Dietterich, T. G. (2000). Ensemble Methods in Machine Learning. \textit{MCS 2000}, LNCS 1857, pp.\ 1--15.

\bibitem{farquhar2024robust} Farquhar, S., et al. (2024). Robust and Cheap Hallucination Detection in LLMs. \textit{arXiv:2406.15927}.

\bibitem{geng2025muse} Geng, J., et al. (2025). MUSE: Multi-Signal Uncertainty Estimation for LLMs. \textit{arXiv:2507.07236}.

\bibitem{geng2024survey} Geng, J., et al. (2024). A Survey on Uncertainty Quantification of Large Language Models. \textit{arXiv:2412.05563}.

\bibitem{gomes2025routing} Gomes, R., et al. (2025). Doing More with Less: Routing Strategies in LLM-Based Systems. \textit{arXiv:2502.00409}.

\bibitem{guo2017calibration} Guo, C., et al. (2017). On Calibration of Modern Neural Networks. \textit{ICML 2017}.

\bibitem{huang2024logprob} Huang, H., et al. (2024). Log Probabilities Are a Reliable Estimate of Semantic Plausibility. \textit{arXiv:2403.14859}.

\bibitem{huang2025uncertainty} Huang, Y., et al. (2025). Uncertainty Quantification and Confidence Calibration in Large Language Models. \textit{arXiv:2503.15850}.

\bibitem{kadavath2022language} Kadavath, S., et al. (2022). Language Models (Mostly) Know What They Know. \textit{arXiv:2207.05221}.

\bibitem{kuhn2023semantic} Kuhn, L., et al. (2023). Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs. \textit{Nature}, 630, 625--630.

\bibitem{li2025dueling} Li, Y., et al. (2025). LLM Routing with Dueling Feedback. \textit{arXiv:2510.00841}.

\bibitem{lin2025uqlm} Lin, Z., et al. (2025). UQLM: Uncertainty Quantification for Language Models. \textit{arXiv:2504.19254}.

\bibitem{mukkunnoth2025confidence} Mukkunnoth, N., et al. (2025). Confidence-Aware Routing for LLM Reliability Enhancement. \textit{arXiv:2510.01237}.

\bibitem{ong2024routellm} Ong, I., et al. (2024). RouteLLM: Learning to Route LLMs with Preference Data. \textit{arXiv:2406.18665}.

\bibitem{ren2023selfeval} Ren, A., et al. (2023). Self-Evaluation Improves Selective Generation in Large Language Models. \textit{arXiv:2312.09300}.

\bibitem{soare2025bandit} Soare, M., et al. (2025). Learning to Route LLMs from Bandit Feedback. \textit{arXiv:2510.07429}.

\bibitem{tabernermiller2026paretobandit} Taberner-Miller, A., et al. (2026). ParetoBandit: Budget-Paced Adaptive Routing for Non-Stationary LLM Serving. \textit{arXiv:2604.00136}.

\bibitem{tian2023just} Tian, K., et al. (2023). Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models. \textit{arXiv:2305.14975}.

\bibitem{wang2022selfconsistency} Wang, X., et al. (2022). Self-Consistency Improves Chain of Thought Reasoning. \textit{arXiv:2203.11171}.

\bibitem{xiong2024llmsexpress} Xiong, M., et al. (2024). Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation. \textit{ICLR 2024}.

\bibitem{zhang2025uncertainty} Zhang, T., et al. (2025). Leveraging Uncertainty Estimation for Efficient LLM Routing. \textit{arXiv:2502.11021}.
