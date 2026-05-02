# Zero-Shot Confidence Estimation for Small LLMs

**When Supervised Baselines Aren't Worth Training**

Code, data, and experiment logs for the paper:

> *Zero-Shot Confidence Estimation for Small LLMs: When Supervised Baselines Aren't Worth Training*

## Key Finding

Average token log-probability — a zero-shot signal requiring no training data — matches or exceeds RouteLLM-style supervised routing baselines in-distribution (AUROC 0.650–0.714 vs. 0.644–0.676 on MMLU-Pro) and substantially outperforms them out-of-distribution (0.717–0.833 vs. 0.512–0.564 on TriviaQA). Supervised routing collapses on a new dataset; zero-shot signals transfer.

## Results at a Glance

| Signal | Qwen-2.5-7B | Llama-3.1-8B | Mistral-7B | TriviaQA (avg) |
|---|---:|---:|---:|---:|
| **logprob (zero-shot)** | **0.714** | **0.650** | **0.678** | **0.782** |
| RouteLLM (supervised) | 0.665 | 0.644 | 0.676 | 0.546 |

---

## Quick Start: Verify Our Results (5 minutes, no GPU needed)

All experiment data is included. You can verify every number in the paper without running any models.

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/zero-shot-llm-confidence.git
cd zero-shot-llm-confidence
pip install -e '.[dev]'

# 2. Run the analysis on pre-computed data
python -m benchmarks.ablation_analysis \
    --run-id v0.1-qwen-20260427 \
    --db-path results/experiment_results.db \
    --output-dir results/verify

# 3. Check the headline number
cat results/verify/v0.1-qwen-20260427/MEMO.md | head -20

# 4. Run all tests
pytest tests/ -q

# 5. Regenerate paper figures
python paper/generate_figures.py
```

This reproduces Table 3, Table 5, and Figures 1–3 from the paper using the included SQLite database.

---

## Full Reproduction from Scratch (~$123, ~29 hours)

To rerun all experiments end-to-end, you need Ollama (local models) and AWS Bedrock (cloud models).

### Prerequisites

| Requirement | What | Why |
|---|---|---|
| Python 3.10+ | Runtime | All scripts |
| [Ollama](https://ollama.ai) | Local LLM inference | Runs the 7B models |
| AWS account | Bedrock API access | Cloud model + judge |

### Step 1: Pull local models (~10 minutes)

```bash
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
ollama pull mistral:7b-instruct
ollama pull qllama/bge-large-en-v1.5
```

### Step 2: Install with Bedrock support

```bash
pip install -e '.[dev,bedrock]'
```

Make sure your AWS credentials are configured (`~/.aws/credentials` or environment variables) with access to Bedrock in `us-west-2`.

### Step 3: Run the Qwen experiment (~$35, ~6 hours)

```bash
# Seed the knowledge base, train RouteLLM baselines, and run the evaluation
python -m benchmarks.ablation_experiment \
    --run-id v0.1-qwen-$(date +%Y%m%d) \
    --local-model qwen2.5:7b \
    --n-seed 1000 --n-eval 1000 --n-training 1000 \
    --train-baselines \
    --confirm-cost --cost-threshold-usd 50
```

### Step 4: Run Llama and Mistral (~$35 each, ~7 hours each)

```bash
# Llama
python -m benchmarks.ablation_experiment \
    --run-id v0.1-llama-$(date +%Y%m%d) \
    --local-model llama3.1:8b \
    --n-seed 1000 --n-eval 1000 --n-training 1000 \
    --output-dir results/v0.1-llama \
    --train-baselines \
    --confirm-cost --cost-threshold-usd 50

# Mistral
python -m benchmarks.ablation_experiment \
    --run-id v0.1-mistral-$(date +%Y%m%d) \
    --local-model mistral:7b-instruct \
    --n-seed 1000 --n-eval 1000 --n-training 1000 \
    --output-dir results/v0.1-mistral \
    --train-baselines \
    --confirm-cost --cost-threshold-usd 50
```

### Step 5: Run the analysis

```bash
# Per-model analysis
for run_id in v0.1-qwen-* v0.1-llama-* v0.1-mistral-*; do
    python -m benchmarks.ablation_analysis --run-id "$run_id"
done
```

### Step 6: TriviaQA cross-dataset check (~$8, ~1.5 hours)

```bash
python -m benchmarks.triviaqa_check --local-model qwen2.5:7b --n-queries 500
python -m benchmarks.triviaqa_check --local-model llama3.1:8b --n-queries 500
python -m benchmarks.triviaqa_check --local-model mistral:7b-instruct --n-queries 500
```

---

## Repository Structure

```
paper/
├── main.tex                 LaTeX source
├── main.pdf                 Compiled paper
├── references.bib           BibTeX references (24 entries)
├── generate_figures.py      Reproducible figure generation
└── figures/                 3 PDF + PNG figures

autodidact/                  Core library
├── types.py                 Data types (Pydantic + dataclass)
├── database.py              SQLite schema
├── llm_client.py            Ollama + Bedrock + OpenAI client
├── knowledge_store.py       FAISS-backed retrieval + Ebbinghaus decay
├── confidence_evaluator.py  5-signal Thompson Sampling evaluator
└── signals/
    └── grounded_self_assessment.py   YES/NO self-assessment probe

benchmarks/                  Experiment scripts
├── ablation_experiment.py   Main experiment harness
├── ablation_analysis.py     AUROC computation + plots + memo
├── datasets.py              MMLU-Pro + TriviaQA loaders
├── labeling.py              Regex + LLM-judge correctness labeling
├── seeding.py               Knowledge base seeding
├── routellm_baseline.py     RouteLLM classifier training
├── routellm_learning_curve.py   Learning curve analysis (Table 5)
├── routellm_cross_model_transfer.py  Cross-model transfer (Section 7b)
├── cross_model_analysis.py  Cross-model comparison tables
├── triviaqa_check.py        TriviaQA evaluation (Table 4)
├── triviaqa_with_retrieval.py  TriviaQA + KB (Section 5)
├── gsa_retrieval_rerun.py   GSA v3 vs v2 comparison (Table 8)
└── label_noise_audit.py     Label noise bound (Section 3.3)

results/                     Pre-computed experiment data
├── experiment_results.db    SQLite: 3,050 rows × 3 models (~7MB)
├── triviaqa_experiment.db   SQLite: TriviaQA results (~2MB)
├── v0.1-qwen-20260427/     Per-model summaries + learning curves
├── v0.1-llama-20260428/
├── v0.1-mistral-20260429/
├── triviaqa_check*/         TriviaQA per-model summaries
├── EXPERIMENT_LOG.md         11 experiments with hypothesis-before-results
└── LAB_NOTES.md              19 debugging entries

tests/                       55 unit tests (run in <3s, no network)
scripts/                     Shell helpers for running experiments
```

---

## Mapping Paper Sections to Code

| Paper section | Script / data |
|---|---|
| Table 3 (per-signal AUROC) | `benchmarks/ablation_analysis.py` → `results/*/summary.json` |
| Table 4 (cross-dataset) | `benchmarks/triviaqa_check.py` → `results/triviaqa_check*/summary.json` |
| Table 5 (learning curve) | `benchmarks/routellm_learning_curve.py` → `results/*/routellm_learning_curve.json` |
| Table 6 (latency) | Computed from `experiment_results.db` latency columns |
| Table 7 (fusion) | `benchmarks/ablation_analysis.py` combo results in `summary.json` |
| Table 8 (GSA v3 vs v2) | `benchmarks/gsa_retrieval_rerun.py` |
| Figure 1 (cross-dataset) | `paper/generate_figures.py` → `fig1_cross_dataset.pdf` |
| Figure 2 (learning curve) | `paper/generate_figures.py` → `fig2_learning_curve.pdf` |
| Figure 3 (signal comparison) | `paper/generate_figures.py` → `fig3_signal_comparison.pdf` |
| Section 3.3 (label noise) | `benchmarks/label_noise_audit.py` → `results/label_noise_audit.json` |

---

## Citation

```bibtex
@article{anonymous2026zeroshot,
  title={Zero-Shot Confidence Estimation for Small LLMs: When Supervised Baselines Aren't Worth Training},
  author={Anonymous},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT
