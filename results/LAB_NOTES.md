# Autodidact v0.1 — Lab Notes

A running log of problems encountered during the ablation experiment and how we fixed them. The point isn't to be polished — it's to capture what actually happened so future-us (and future readers) don't have to rediscover it. Especially useful for:

- Debugging a rerun that hits the same issue.
- A blog post or paper appendix that's honest about what broke and why.
- Onboarding new contributors who will hit some subset of the same potholes.

Every entry has:
- **What we saw** (the surface symptom)
- **Real cause** (what was actually wrong)
- **Why we missed it** (the diagnostic hole)
- **Fix** (code pointer + summary)
- **Lesson** (what to do differently next time)

## P1. INSERT schema drift — "37 values for 34 columns"

**What we saw.** Every query in the dry-run failed at the final DB write with `37 values for 34 columns`. The wrapper error handler (`_record_failure_row`) also failed with the same message, so the experiment produced 0 completed rows and no failure rows either. 40 consecutive queries failed in the same 20 seconds.

**Real cause.** When we added the `gsa_extraction_mode` column (Fix A, to persist how each grounded-self-assessment signal was derived), we only partially updated the INSERT statements:
- The column list got one new entry — correct.
- The tuple of Python values got one new item — correct.
- **The `VALUES (?, ?, ...)` placeholder string was not updated — it kept three extra `?` leftover from an earlier edit.**

Both `_process_query` and `_record_failure_row` had the same bug, so the failure handler couldn't even record the failure. A bug in the code reporting other bugs hid everything behind the same misleading message.

**Why we missed it.** We ran the 25-test unit suite after adding the column, but none of the tests exercised the INSERT path end-to-end — the existing tests use `KnowledgeStore` not the raw `experiment_results` INSERT. The fix-A change shipped green and blew up only on real run.

**Fix.** `benchmarks/ablation_experiment.py` — corrected both VALUES placeholder strings to match the 34-column tuple. Added a post-fix end-to-end smoke that constructs a mocked `_process_query` invocation and asserts one row was written with all 34 columns populated correctly. That smoke now catches this regression in under 2 seconds.

**Lesson.** Any time you add a column to a table with an INSERT statement, the three things (column list, placeholders, values tuple) must change in lockstep. Separately, the failure-reporting path needs its own unit test because bugs there will mask everything upstream. A simple defensive pattern:

```python
cols = "col1, col2, col3, ...".split(",")
placeholders = ",".join("?" for _ in cols)
sql = f"INSERT INTO t ({','.join(cols)}) VALUES ({placeholders})"
# Now column count, placeholder count, and tuple length are the same variable.
```

Small refactor, but it eliminates this entire class of bug.

## P2. best_combo_name parsing — trailing whitespace KeyError

**What we saw.** The analysis step (step 3) crashed with `KeyError: 'all_six_mean '` (note the trailing space). The experiment itself had completed successfully — 50 of 50 queries written.

**Real cause.** The headline `best_combo` is formatted as `"all_six_mean (mean)"`. To look up the combo's signal list, we were splitting on `"("` and indexing `[0]`, which returned `"all_six_mean "` (with trailing space). That key didn't exist in the `COMBOS` dict.

Elsewhere in the same file we correctly split on `" ("` (with the space), which gives clean `"all_six_mean"`. The inconsistency was a copy-paste error.

**Why we missed it.** The retrieval-stratification block only runs when `retrieval_recall_at_5.sum() > 0`. Our offline smoke tests used synthetic data with `retrieval_recall_at_5 = 0` for every row, so this code path was never exercised. Real data had non-zero recall, triggered the branch, and crashed.

**Fix.** `benchmarks/ablation_analysis.py` — switched the split to `" ("` to match the canonical form. Added an `" (" in best_combo_name` guard to defend against malformed names. Smoke tests updated to populate `retrieval_recall_at_5 > 0` for at least some rows so this code path is covered.

**Lesson.** Two lessons worth noting:

1. Synthetic smoke test data needs to exercise every code path, including the ones gated on "real data looks like X." Our synthetic data was too clean.
2. When a string parses into parts, use the same separator consistently. We should have a single helper `_parse_combo_name(s) -> (key, variant)` rather than inlining `.split(...)` at each call site. Defined-once is harder to get wrong than defined-twice.

## P3. Ollama logprobs silently ignored

**What we saw.** After fixing P1 and P2, the dry-run completed — but three signals had AUROC exactly 0.500 with zero-width confidence intervals:

| Signal | AUROC | 95% CI |
|---|---|---|
| `grounded_self_assessment` | 0.500 | [0.500, 0.500] |
| `logprob_uncertainty` | 0.500 | [0.500, 0.500] |
| `knowledge_similarity` | 0.484 | [0.448, 0.500] |

An AUROC of 0.500 with a zero-width CI is not a weak signal — it's a **constant signal**. Every query produced the same value.

**Real cause.** Two separate problems, stacked:

1. **Our client called Ollama with the wrong parameter names for logprobs.** We passed `options.include_logprobs = true` inside the `options` dict. Ollama 0.12.11+ uses a **top-level** `logprobs: true` boolean and `top_logprobs: N` integer on the request body, not inside `options`. Older Ollama versions (<0.12.11) don't have logprob support at all and silently discard unknown parameters.

   Result: Ollama always returned the chat response without any logprob fields. Our client's parser saw no `logprobs` key, degraded gracefully by returning `avg_logprob=None` and `top_logprobs_by_position=[]`. Downstream, the GSA signal fell through all three fallback tiers — logprob softmax failed (no logprobs), text hard failed for inscrutable single-token responses, so it returned the neutral 0.5 — and `logprob_uncertainty` also got its neutral fallback of 0.5.

2. **Retrieval density was near-zero.** The 100-query seed across 14 MMLU-Pro categories left `retrieval_recall_at_5 ≈ 0.02` — only 1 of 50 eval queries had any in-category retrieved hit. With knowledge-similarity at 0 (below the 0.75 threshold) for ~98% of queries, `knowledge_similarity` was a near-constant signal.

**Why we missed it.** Three reasons:

1. **We didn't verify logprobs end-to-end against a live Ollama before running.** Our unit tests mocked the HTTP response, which meant the test data had logprobs in it. Real Ollama, we'd assumed, behaved like our mock. It didn't.

2. **Our graceful degrade path was too graceful.** The client didn't emit a warning when logprobs were absent — it just returned empty. For testing and mock purposes this is the right design, but for a long-running experiment against a real backend it obscured a real bug. We needed either (a) a startup probe that warns "this Ollama version doesn't expose logprobs, X signals will be degraded," or (b) a post-run health check that flags "0% of queries had non-empty logprobs."

3. **We estimated retrieval density casually.** We did not run even a single MMLU-Pro query through retrieval before committing to seed=100. A 5-minute sanity check would have revealed retrieval_recall was near-zero and prompted a larger seed.

**Fix.**

- `autodidact/llm_client.py` — moved `logprobs: true` and `top_logprobs: N` to the top level of the request body. Updated the response parser to look for `data.logprobs` (the new top-level field) before falling back to `message.logprobs` (the old, never-populated nested field). Tested against live Ollama 0.21.1 and confirmed we now extract real token probabilities.
- Will add an **experiment preflight** that fires a 1-token "YES/NO" probe through the configured local client and verifies `top_logprobs_by_position[0]` is populated. If it isn't, abort before spending hours and dollars.
- For retrieval: running next dry-run with N_SEED=500 to verify recall improves before committing to the real run.

**Lesson.** Graceful degrade is correct for individual per-query failures, but at the experiment level we want loud-fail or loud-warn. Three specific principles:

1. **Probe before committing.** Before any long-running experiment, probe every critical signal-producing path with a single canary query and assert it produces non-trivial output. A 30-second probe saves 4 hours and $50.

2. **Version pin the probe.** Document which Ollama/Bedrock/model versions we tested against. When a user runs against a different version, the probe either passes (great) or fails loudly (great). Silent degradation is the worst case.

3. **Health metrics in the run summary.** The memo should surface "what fraction of signals were computed from real data vs. fallback" at the top, not buried in per-signal AUROC tables. A dead signal should be loud.

## P4. RouteLLM baselines trained under broken conditions

**What we saw.** `routellm_no_memory` and `routellm_plus_ks` both achieved AUROC 0.683 with **identical** CIs. They should be different — the plus_ks variant has the knowledge-similarity feature as extra input, and a sensible classifier should do something with it.

**Real cause.** When baselines were trained (before we knew about P3), `knowledge_similarity` was essentially always 0 for training queries. That means the plus_ks classifier had a feature column that was constant at 0 for ~99% of training rows. Logistic regression correctly learned "this feature carries no information, ignore it." The two classifiers converged to the same decision boundary because they saw essentially the same training data.

**Why we missed it.** We didn't inspect the training data summary statistics after baseline training. A single `SELECT AVG(knowledge_similarity) FROM routellm_training_rows` would have revealed the issue. Instead we assumed the baselines were well-formed and moved on.

**Fix.** For the real run, we'll retrain baselines with N_SEED=500 (higher retrieval density) and N_TRAINING=1000. Also adding a printout at the end of baseline training that shows feature statistics so this is caught before the main experiment.

**Lesson.** Machine learning pipelines need data-quality assertions at every stage boundary. After loading data, after feature engineering, after training — each should emit "here's what we fed in, here's what looked weird" output that a human can eyeball in 3 seconds.

## P5. Small model refuses to answer (honest but unhelpful)

**What we saw.** Even after the text-based fallback tier successfully parsed `qwen2.5:7b`'s YES/NO token for GSA, `avg_gsa = 0.0` across all 50 queries. The model said NO to every single "do you have enough information?" probe.

**Real cause.** With retrieval density near zero (P3), GSA was asking the model "do you have enough information?" with essentially no retrieved context shown in the prompt. The honest answer for a 7B model seeing a hard MMLU-Pro question with no memory is "NO." The model was being accurately uncertain; the problem was we had nothing to give it.

**Why we missed it.** We designed GSA to be "grounded on retrieval" but hadn't stress-tested what it does when retrieval returns nothing. The signal is only meaningful when the retrieved context is at least sometimes relevant.

**Fix.** Two things, both partial:

- **Bigger seed store.** N_SEED=500 should give more queries actual retrieved content in the prompt, letting GSA differentiate.
- **Prompt engineering follow-up.** If the model persistently refuses even with retrieval available, we may need to reframe the prompt ("is the retrieved content sufficient to answer?" vs "do you know?"). This is a v0.1.1 item.

**Lesson.** A confidence signal that depends on retrieval can't be evaluated in isolation — its usefulness is bounded by the retrieval quality. When you design a dependent signal, always evaluate it across a range of dependency quality (good retrieval vs. bad retrieval) or you're not measuring the signal, you're measuring the compound system.

## Meta-observations

**Error cascades hide causes.** We saw three cascading errors in this run:
1. INSERT schema drift (P1) masked per-query failures.
2. Failure handler bug (same P1) masked the upstream error text.
3. The parse error (P2) masked the fact that the experiment had actually succeeded.

Each surface error pointed away from its root cause. Debugging required peeling back one layer at a time. Lesson: **every error handler is also a potential error source**. Test them.

**Synthetic data is a lie.** Our unit and smoke tests all passed after every fix, but real Ollama + real Bedrock + real MMLU-Pro produced a cascade of failures our synthetic tests didn't catch. Three things were wrong that tests didn't know to check:

1. Ollama's actual response format (P3).
2. `retrieval_recall` in our real setup (P3/P4).
3. Small model's actual YES/NO behavior (P5).

Synthetic data tested that our code produces the right output when the backend cooperates. But the interesting failures come from backends that don't cooperate. **There's no substitute for a small live probe against real infrastructure.**

**Fallbacks should be auditable.** Our GSA signal has three fallback tiers (logprob softmax → text hard → neutral 0.5). Each tier is correct in isolation. But across 50 queries, 100% of rows ended in tier 2, which is "all three tiers failed their primary purpose." We had no mechanism to detect this at the experiment level — it only showed up as suspicious AUROCs. **A signal-health summary that reports "tier 1: 0%, tier 2: 100%, tier 3: 0%" should be in every memo**, prominently.

**$5-ish dry runs earn their cost many times over.** Every failure above (except P5, which is interpretive not technical) was caught by the dry-run at sub-$5 cost. If we'd gone straight to the real $52 run we'd have burned everything and spent another day resolving the same issues. The dry-run ritual is non-negotiable.

## Reusable checklist for the next big run

Before running a multi-hour / multi-dollar experiment, verify:

- [ ] **Canary probe** — single 1-token query through the configured local client. Assert `top_logprobs_by_position[0]` is populated with at least two distinct tokens.
- [ ] **Canary probe** — single call through the cloud client. Assert non-empty content.
- [ ] **Retrieval density** — sample 20 eval queries, measure `retrieval_recall_at_5`. If < 0.15, bump `N_SEED`.
- [ ] **Baseline training data sanity** — after training RouteLLM baselines, print `{count, mean, std}` of each feature column. Flag any feature whose std is below 0.05.
- [ ] **Dry-run end-to-end** — at least 20 seed + 50 eval queries. Inspect the memo. Every signal should have non-zero variance.
- [ ] **Signal-health headline in memo** — if any signal shows constant output or extraction-mode distribution is pathological (100% of rows in one fallback tier), loud warning at top of memo, not buried in the ablation table.

When any of these fail, STOP and fix before scaling up.


## P6. --dry-run flag silently overrode user-supplied N_SEED

**What we saw.** We set `N_SEED=500 ./scripts/run_experiment.sh --dry-run`. After the run, only 20 knowledge entries existed in the store. The N_SEED value was silently overridden back to 20 by `--dry-run`.

**Real cause.** The flag-handling in `scripts/run_experiment.sh` unconditionally set `N_SEED=20 N_EVAL=50 N_TRAINING=100` when `--dry-run` was passed, overwriting env-var values the user had already supplied.

**Why we missed it.** We didn't think about env-var-vs-flag precedence when writing the script. The contract should be "user-supplied values win" but the implementation was "--dry-run always wins."

**Fix.** Record whether the user explicitly set each N_* env var *before* applying defaults. In the `--dry-run` branch, only override vars the user didn't set. See `scripts/run_experiment.sh`.

**Lesson.** Flag-vs-env precedence needs an explicit contract. The pattern that works for shell: capture `USER_SET_X="${X+yes}"` *before* `X="${X:-default}"`, then flags that want to override conditionally check `[ -z "$USER_SET_X" ]`.

## P7. GSA signal inverted — model says "NO" to things it actually can answer

**What we saw.** After the logprob fix (P3), GSA had real variance but its AUROC was 0.183 (well below 0.5 = chance). An AUROC of 0.183 means the signal is "correct but inverted" — when it says p_yes is high, the local model is actually wrong; when p_yes is low, the local model is actually right. Meanwhile `avg(p_yes) = 0.006` — the model says NO to 99% of queries, but `avg(local_correct) = 0.38` — the model gets 38% right. It is systematically pessimistic.

**What this actually means.** The signal CARRIES information — AUROC 0.817 if you flip it — but in the wrong direction for fusion. Three non-mutually-exclusive explanations:

1. **Model-specific humility bias.** Qwen2.5:7b was RLHF'd to express uncertainty; it says NO to "do you have enough info?" as a default. Other models might behave differently.
2. **Prompt framing.** Our prompt asks about *information sufficiency*. A model can answer correctly from its own weights without any retrieved information, but still say "I don't have enough info" because the prompt emphasizes the information-recall framing. The gap between "do I have this fact written down?" and "can I produce a correct answer?" matters.
3. **MMLU-Pro difficulty.** The questions are specifically designed to be hard. The model is correctly uncertain about most of them, but also correctly guessing better-than-chance. Confidence and correctness can dissociate when the base rate is non-trivial.

**Why "just flip it" is a bad fix.** Post-hoc inversion means we've fit to the specific model and prompt in our run. If we ship this and a user swaps to a different local model, the inversion might disappear (or worsen), and nothing in the system would adapt. It would be a silent failure. Auto-detecting inversion at training time and flipping is slightly better, but still hides the underlying cause — we'd be reporting "AUROC 0.817" while the prompt is genuinely asking the wrong question.

**The right fix is to reframe the prompt** so it asks something the model answers calibratedly. Candidate prompts to try:

1. `current`: "Do you have enough information to answer this question correctly?"
2. `direct`: "Can you answer this question correctly?" — what the user suggested. Removes the information-recall framing.
3. `confidence`: "Are you confident you can answer this question?" — softer, measures felt confidence.
4. `prediction`: "Will your answer to this question be correct?" — task-framed prediction.

We'll measure all four on a small (n=30) preflight set and pick the one with the best combination of (a) AUROC magnitude and (b) positive sign, without post-hoc flipping.

**Fix (planned).** Add a prompt selector to `autodidact/signals/grounded_self_assessment.py`. Run a `benchmarks/gsa_prompt_study.py` mini-experiment that tests all variants on a shared 30-query subset, outputs a small table, and picks the winner. Document the winner in the memo so anyone reading knows which prompt was used.

**Lesson.** When a signal has the right magnitude but the wrong sign, *that's diagnostic information about the prompt*, not a bug to patch around. The instinct "auto-flip if AUROC < 0.5" would have shipped a silent model-specific behavior. Always ask: if we swap the model or change the prompt, does this fix still apply? If not, it's a hack.

## P8. `logprob_uncertainty` requires a full local generation — architectural cost

**Observation (not a bug).** The dry-run showed `logprob_uncertainty` as the best single signal at AUROC 0.642. Architecturally, this signal requires the local model to complete a full answer (~200 tokens on average) before the signal can be computed. Meanwhile, GSA is a 1-token generation — ~200x cheaper.

**The tradeoff for a production system.** If we route a query to cloud because the confidence evaluator said "escalate," we just paid for the local generation we're about to discard. Over a corpus, that's meaningful wasted compute.

**Implication for the memo.** When interpreting per-signal AUROC, we need to also report cost-per-signal. A cheap signal with lower AUROC may still win the product-relevant Pareto frontier against an expensive signal with higher AUROC. The current memo doesn't surface this; it treats all signals as cost-equivalent.

**Fix (planned).** Add a "latency-per-signal" column to the per-signal AUROC table in the memo, populated from the already-recorded `latency_*` columns in `experiment_results`.

**Lesson.** AUROC alone doesn't pick the right signal for a product. For routing specifically, you want the best AUROC *among signals cheap enough that computing them is worthwhile versus just going to cloud directly*. Flag this in the memo so readers don't over-index on the raw AUROC table.


## P9. Retrieval injection HURTS grounded self-assessment (counterintuitive)

**What we saw.** A prompt ablation study (3 studies × 4 prompt variants × 30 queries, n=360 total probes) against a 500-entry knowledge store showed:

- Without KB context:           `confidence` prompt AUROC **0.620** (winner)
- With top-5 injected, no filter:  `confidence` prompt AUROC 0.538
- With top-5 injected, ≥0.7 score: `confidence` prompt AUROC 0.538

Higher is better. Injecting retrieved knowledge into the GSA prompt **LOWERED the AUROC** by 0.082. Counterintuitive — the whole design premise was that retrieved context would ground the self-assessment.

**Per-query inspection reveals why.** On the 5 queries (of 30) where retrieval actually returned hits above 0.75 similarity:

| Query topic | Local correct? | GSA with KB | GSA without KB |
|---|---|---|---|
| Scarcity definition (economics) | YES | 0.000 | **1.000** |
| Oxygen cylinder calc (engineering) | YES | 0.000 | **0.651** |
| Colonial letter (history) | NO | 0.727 | 0.985 |
| Truth table (philosophy) | NO | 0.980 | 1.000 |
| Loan interest (business) | NO | 0.601 | 0.400 |

On both queries the local model actually got right, retrieval injection made it say "NO I can't answer" (with p_yes near 0). The retrieved content confused the self-assessment into rejecting answers the model would have otherwise given correctly.

**Why this happens (hypothesis).** The GSA prompt primes the model to think about "what do I know?" When the prompt shows retrieved content, the model shifts attention from its own knowledge to evaluating the retrieved content. If the retrieved content doesn't directly contain the answer, the model concludes "I don't see the answer here" and says NO — even if it could have answered from its own weights.

This is a specific instance of a broader retrieval-augmentation failure mode: **retrieval can hurt when the model has good intrinsic knowledge AND the retrieved content is tangentially-relevant (not directly answer-containing).**

**Why we missed it.** The v1 GSA design was reasoned from first principles ("ground the confidence signal on actual retrieval"), not measured. The ablation study was cheap (~$1.50) and immediately revealed the problem.

**Fix.** Updated `autodidact/signals/grounded_self_assessment.py`:
- Prompt changed to the `confidence` variant (no retrieval injection): "Are you confident you can answer this question correctly? Respond with exactly one token: YES or NO."
- `retrieved_hits` parameter retained for API stability but silently ignored.
- Renamed class from `GroundedSelfAssessment` → `SelfAssessment` (backwards-compat alias kept).
- Prompt version bumped to `gsa-v2-confidence` so runs are distinguishable in stored data.

**Lesson.** When your architecture claims "X should help Y," measure it. Don't assume. The claim was "retrieval should ground self-assessment" and it turned out to actively hurt. Every "should help" in the design deserves an ablation before shipping.

## P10. Study-A and Study-C produced identical results — threshold applied at wrong layer

**What we saw.** Study A (KB, threshold=0.0) and Study C (KB, threshold=0.7) produced identical per-variant AUROC numbers, despite intending different retrieval behavior.

**Real cause.** The `KnowledgeStore.search()` method has its own internal similarity floor (default 0.75) hardcoded in `_faiss_search`:

```python
if idx < 0 or sim < self.config.similarity_threshold:  # 0.75
    continue
```

The `--kb-threshold` flag on the prompt study was applied AFTER the store had already filtered at 0.75. So:
- Study A (`threshold=0.0`): store returned hits ≥ 0.75; study filter of 0.0 added nothing. 5/30 queries had hits.
- Study C (`threshold=0.7`): store returned hits ≥ 0.75; study filter of 0.7 still admitted all of them. 5/30 queries had hits.

Same 5 queries in both, same hits, same model responses. Identical AUROC.

**Why we missed it.** Two-layer filtering without a clean abstraction. The KnowledgeStore's internal threshold is a design decision (always filter below this for "memory relevance") while the study's threshold was intended as an experimental knob. We didn't notice they compose until we saw identical output.

**Fix.** For future retrieval-quality studies, call `_filtered_search()` directly with no min threshold, or add an override parameter to `search()`. For the GSA prompt study specifically this doesn't matter anymore (P9 says drop retrieval injection altogether), but documenting so the next retrieval-quality experiment avoids the trap.

**Lesson.** Thresholds composed across library and experiment layers produce non-obvious behavior. Either make the library layer's threshold opt-in (caller must pass it), or document composition explicitly. A reasonable rule: library defaults should be loose, experiments should specify.

## P11. Architecture decision: stop injecting retrieval into the GSA prompt. Also test whether retrieval helps the MAIN answer (open question).

**What we decided.** Per P9 data:
- GSA no longer injects retrieved hits. Prompt is `confidence`, no context.
- The main experiment harness's `_process_query` still injects retrieved hits into the FULL answer prompt (the one that produces `logprob_uncertainty`). **We haven't tested whether that helps or hurts.**

**Why this matters for v0.1.** Our logprob_uncertainty AUROC in the earlier dry run was 0.642 — our current best signal. But that number was computed from answers generated WITH retrieval injected. If retrieval injection is hurting answer quality (as it hurt GSA), we might be leaving AUROC on the table, and the real answer would be to strip retrieval from the main prompt too.

**New study planned (answer_quality_study.py).** Run n=60 queries. For each, generate TWO local answers: one with retrieval injected, one without. Label both via the same cloud-judge pipeline. Report:
- Accuracy with KB vs without KB
- Per-query flips (correct→wrong, wrong→correct)
- McNemar two-sided p-value

Decision rule:
- Delta ≥ +5% and p < 0.05 → retrieval helps, keep it
- Delta ≤ -5% and p < 0.05 → retrieval hurts, strip it
- |Delta| < 3% → wash, drop for latency savings

**Lesson.** Each use of retrieved knowledge is a separate design decision. "Retrieval for the signal" and "retrieval for the answer" are different bets and need to be tested independently. Don't generalize from one to the other.


## P12. Signal generalization is an open question — cross-model sweep added to the plan

**What we realized.** Everything we've measured so far is "on `qwen2.5:7b` with MMLU-Pro." The findings are empirical hacks, not theory. If we ship v0.1 with signals chosen on one model, we have no evidence they work on `llama3.1:8b`, `mistral:7b`, or any other local model. Our product claim — "works for any local model" — is not supported.

**Theoretical audit of each signal (summary from chat):**

| Signal | Published foundation | Generalization prediction |
|---|---|---|
| Logprob uncertainty | Guo et al. 2017 (calibration); Kadavath et al. 2022 | Should generalize to any model with calibrated token distributions. Better on base models than heavily-RLHF'd. |
| Self-consistency | Wang et al. 2022 | Should generalize to any model with non-degenerate T>0 sampling. Requires 5+ samples for robust signal; we use 2. |
| Thompson fusion | Thompson 1933; Agrawal & Goyal 2012 | Algorithm itself is model-agnostic. Only as good as input signals. |
| Energy scorer | Standard supervised learning | Generalizes; AUROC depends on model-specific competence patterns. |
| Knowledge similarity | IR / retrieval literature | Bounded by embedding model quality, not local LLM. |
| Query classification | None (hand-coded heuristic) | Unknown. |
| Self-assessment (confidence prompt) | Kadavath et al. 2022 | Thin. Depends heavily on RLHF program and model size. Winning prompt may be model-specific. |

**What the data could support (if we run it):**
- Cross-model study: replicate the ablation on `qwen2.5:7b`, `llama3.1:8b`, `mistral:7b`.
- If the same signals win across 3 models → strong evidence of generality.
- If different signals win per model → the paper becomes "framework for per-model adaptive confidence" (more honest) rather than "universal signal stack" (overclaim).

**Why it matters.** Without cross-model evidence, shipping v0.1 means shipping "something that worked once on one model." First user trying a different local model gets garbage results and posts a bad review. Conversely, documenting per-model variation is a MUCH stronger paper — it shows we care about generality.

**Decision.** Add cross-model sweep to the plan. Run the current commitments first (answer-quality study, n=100 GSA replication) then do the cross-model replication on the top 2-3 signal configurations across 3 local models. Cost: ~3× current budget. Time: ~3×.

**Lesson.** Empirical hacking produces publishable papers only when it comes with a generalization claim backed by evidence. "It works on X" is a case study. "It works on X, Y, Z with theoretical explanation why" is a contribution. We were trending toward the former. Fixing course.


## P13. v0.2 research framing — the actual novelty is the non-stationary growing-arm bandit

**Observation.** v0.1 as specified (static-memory ablation on one model) is a necessary precondition but NOT the novel contribution we care about. The real novelty — the thing no prior routing paper addresses — is:

- Local model is capability-constrained
- Its knowledge base GROWS via cloud escalation, making the local arm non-stationary
- Confidence evaluation must track this shift over time
- Optimal routing threshold itself is a function of memory size N

**Why prior work doesn't cover this.**

- RouteLLM, Adaptive LLM Routing Under Budget Constraints, Learning to Route from Bandit Feedback, LLM Routing with Dueling Feedback — all treat models as FIXED arms. Routing is "pick one of N fixed models per query." The arms don't grow; their capability distribution is stationary.
- Mem0, MemPalace — memory architectures, no routing/confidence story, no bandit framing.
- Kadavath et al., UQLM — confidence estimation, but on single-model settings, no memory-aware framing.

**The proper theoretical framing.** Non-stationary multi-armed bandit. Specifically: one arm's reward distribution drifts monotonically (capability grows) while others stay fixed (cloud models). Relevant theory: Garivier & Moulines (2011, "On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems"), Besbes et al. (2014, "Stochastic Multi-Armed-Bandit Problem with Non-stationary Rewards"), Russo & Van Roy (2014, "Learning to Optimize via Posterior Sampling").

Standard Thompson Sampling assumes stationary rewards. Applying it to our setting, as we currently do in v0.1, is provably sub-optimal in the non-stationary regime. There's theory on how to fix it (discounting, sliding windows, restart policies) but nobody has applied it to LLM routing specifically.

**v0.2 experiments sketched (for later design doc):**

1. **Capability-growth curve.** 2000 sequential MMLU-Pro queries, empty store → grows as we escalate. Rolling local resolution rate + accuracy + cost vs query index.
2. **Per-signal AUROC evolution.** Bin by memory-size-at-query; AUROC per bin. Expect knowledge_similarity to grow with N; expect logprob/self-consistency stable.
3. **Threshold adaptation.** Plot Pareto (local rate, accuracy) at different N. Does the optimal threshold shift predictably?
4. **Non-stationary Thompson.** Compare Static-TS (v0.1) vs NS-TS (exponential discounting) vs Restart-TS. Does non-stationarity fix help?
5. **Escalation-feedback-loop robustness.** Deliberately poison early memory with wrong entries; measure degradation. Tests system robustness.

**v0.1 status re: this framing.** v0.1 is the static-memory precondition: "given a fixed memory, which signals predict local correctness?" Necessary but not sufficient for the novel claim. We need v0.1 done cleanly first, THEN v0.2 builds on it.

**Paper shape IF v0.2 succeeds.** "LLM routing with knowledge-growing arms is a non-stationary bandit problem. We characterize the regret structure, propose and evaluate adaptive routing strategies, and show empirically on MMLU-Pro / BBH / TriviaQA that standard static routing under-performs in this regime by X-Y% at fixed cost budgets." That's a defensible research contribution with clear prior art (NS-bandit literature), clear gap (nobody applied it to LLMs), and a clean empirical claim.

**Lesson / reframing.** We're not building "a better confidence estimator" (crowded space). We're building "a routing system for the non-stationary growing-arm bandit, instantiated for local+cloud LLM routing with escalation-based memory growth" (ungcrowded). The theoretical framing and the product framing align: the product LEGIT requires this framing because memory DOES grow in production, so stationary bandit routing WILL underperform over time.

This is the strongest v0.2 pitch I can see. v0.1 stays the same; v0.2 spec gets this scope.


## P14. Cross-model GSA results: winning prompt is model-specific. Single-prompt-for-all is falsified.

**What we measured.** Same 4-prompt GSA ablation (current, direct, confidence, prediction) on 3 local models (qwen2.5:7b, llama3.1:8b, mistral:7b-instruct), n=100 queries each, same dataset seed, no KB injection. Total ~$6 across the three studies.

**Results.**

| Model | current | direct | confidence | prediction | Winner | Max AUROC |
|---|---|---|---|---|---|---|
| qwen2.5:7b | 0.470 | 0.591 | **0.636** | 0.525 | `confidence` | 0.636 |
| llama3.1:8b | **0.545** | 0.535 | 0.519 | 0.515 | `current`* | 0.545 |
| mistral:7b-instruct | 0.562 | **0.747** | 0.684 | 0.657 | `direct` | 0.747 |

\* "Winner" on Llama is barely above chance (0.545 vs 0.515 worst). All four Llama variants are within 0.03 AUROC, so the "winner" label is cosmetic. Given the ~±0.10 bootstrap CI at n=100, none of Llama's variants is statistically distinguishable from chance.

**Three clear findings.**

1. **Single-prompt-for-all is falsified.** The winning prompt differs across models (confidence / current / direct). Any v0.1 architecture decision that assumed one-prompt-fits-all is wrong.

2. **Llama-3.1-8B has essentially no self-assessment signal.** All four variants hover around chance. The model's Y/N response to "are you confident?" is uncorrelated with whether it's actually correct.

3. **Mistral-7B has the strongest self-assessment signal** among the three. AUROC 0.747 with the `direct` prompt is the best cross-model number we've measured. Worth noting for product: if self-assessment matters for the deployed use case, Mistral is a better local-model choice than Qwen or Llama.

**Theoretical interpretation (hypothesis, not proof).** Per [Kadavath et al. 2022](https://arxiv.org/abs/2207.05221), calibrated self-assessment depends on RLHF programs that specifically reward honest uncertainty reporting. Qwen2.5's public documentation cites self-critique data in postraining. Mistral's approach is different and apparently produces well-calibrated self-confidence for this query type. Llama3.1's RLHF was optimized for helpfulness and harmlessness — not for calibration specifically. The results are consistent with this theoretical framing; they don't prove it.

**Why it matters for v0.1.**

The `confidence` prompt choice made earlier (based on qwen2.5:7b only) does NOT generalize. The v0.1 claim "we validated GSA as a signal on small local models" has to become one of:

- **Option A.** "GSA works on some models but not others. Product ships with per-model auto-probe on install." This is the honest path. Cost: 10 minutes and $0.50 per install to run the mini prompt study.
- **Option B.** "GSA is too unreliable across models. Drop it from v0.1." Weaker signal stack, but simpler story.
- **Option C.** "v0.1 scopes claims to specific validated models. Future work covers more." Also honest, but narrow.

**Why it matters for v0.2.** The v0.2 non-stationary-bandit framing doesn't depend on GSA generalizing — it depends on AT LEAST ONE signal working per model. Logprob_uncertainty is the more likely universal signal (theoretical grounding via Guo et al. calibration), and we haven't yet measured it cross-model. If logprob_uncertainty generalizes and GSA doesn't, v0.2 can proceed with logprob_uncertainty as the backbone signal.

**Falsifiable predictions this result generates.**

- **Prediction 1:** On models trained specifically with self-critique or constitutional AI (e.g. Anthropic's Claude, DeepSeek, any with explicit calibration loss), self-assessment signals should work well.
- **Prediction 2:** On models trained for narrow helpfulness without calibration rewards (Llama3.1), self-assessment should be weak regardless of prompt.
- **Prediction 3:** The yes_rate under each prompt variant is a proxy for the model's bias. Models with yes_rate near 0.5 across variants are likely well-calibrated; models with extreme yes_rate (near 0 or 1) are biased. Llama3.1 yes_rate across variants: 0.030 / 0.100 / 0.240 / 0.260 — all low, confirming it's biased-pessimistic.

**Lesson.** A claim validated on one model is a case study, not a generalization. Before this cross-model study we would have shipped "confidence is the right prompt." That would have been wrong for 2/3 of users. The cross-model sweep cost $6 and 2 hours and saved us from a silently broken shipping default. Every architectural claim in v0.1 deserves this treatment before it's locked in.


## P15. Cross-model answer-quality results: retrieval injection is a wash, but retrieval density is the real bottleneck

**What we measured.** Same answer-quality study (n=60 queries, with-KB vs without-KB local generation) across 3 local models, using the same 500-entry KB and same embedding model.

**Results.**

| Model | Acc WITH KB | Acc WITHOUT KB | Delta | McNemar p | n_with_hits/n |
|---|---|---|---|---|---|
| qwen2.5:7b | 0.417 | 0.400 | +0.017 | 1.000 | 10/60 |
| llama3.1:8b | 0.450 | 0.417 | +0.033 | 0.625 | 10/60 |
| mistral:7b-instruct | 0.267 | 0.283 | -0.017 | 1.000 | 10/60 |

**Four findings.**

1. **Retrieval injection is statistically a wash on all three models.** No p-value reaches 0.05. Deltas range from -0.017 to +0.033. At n=60 with only 10 queries actually retrieving anything, we are underpowered to detect any effect smaller than about ±0.08.

2. **The direction varies by model.** Qwen and Llama hint at retrieval help; Mistral hints at retrieval hurt. Given the noise these are not interpretable individually, but the lack of directional consistency says retrieval injection does not reliably help.

3. **Mistral's base local accuracy is notably worse (0.28) than Qwen's (0.40) or Llama's (0.42) on MMLU-Pro.** That's a pure model-capability finding unrelated to retrieval. For a product, base local quality matters: a weaker local model escalates more often.

4. **Retrieval density is identical at 10/60 across all three models.** This is because retrieval depends on the embedding model and the queries, not the local LLM. With only 17% of queries retrieving anything, the answer-quality study is effectively measuring on n=10 — dramatically underpowered.

**The real bottleneck: retrieval pipeline, not the local LLM.**

The signal we care about most (knowledge_similarity) and the retrieval-injection question are both limited by the same underlying problem: 83% of eval queries retrieve nothing relevant from a 500-entry KB. Until we fix this, we can't cleanly measure whether retrieval helps the main answer.

Possible fixes (in rough order of effort):
- Replace `nomic-embed-text` with `bge-large-en-v1.5` or `mxbai-embed-large-v1` — better general-QA retrieval, same latency budget. 1-hour change.
- Add a reranker (e.g. `bge-reranker-base`) for top-20-then-rerank-to-5. 2-hour change, ~100ms latency per query.
- Query expansion (HyDE: embed a hypothetical answer, not just the question). Requires one extra local LLM call per retrieval. Meaningful work.
- Grow the KB substantially (current: 500 entries). v0.2's sequential experiment naturally grows the KB as it goes, partially addressing this organically.

**Decision for v0.1.**

- **Drop retrieval injection from the main answer prompt.** Evidence: wash across 3 models at current pipeline quality. Saves ~20% of inference latency per query. If a user complains about accuracy, we can measure again with improved retrieval.
- **Keep `knowledge_similarity` in the ablation.** It has theoretical grounding, it's cheap to compute, and v0.2 will re-test under growing-memory conditions where it's expected to matter more.

**Decision for v0.2.**

- **Improve retrieval pipeline BEFORE the non-stationary experiment.** Running the growing-memory experiment with weak retrieval will just measure weak retrieval. The v0.2 spec should include a retrieval upgrade as a precondition.
- **Measure retrieval density as a function of N.** We expect retrieval_recall_at_5 to grow with KB size. At N=5000 with a better embedder it should be in the 40-60% range. That's when the knowledge_similarity signal should come alive.

**Lesson.** When a signal has AUROC near chance, check the pipeline quality before concluding "signal doesn't work." In our case, retrieval quality confounded three separate measurements (knowledge_similarity AUROC, GSA-with-KB AUROC, retrieval-injection-into-answer effect). One pipeline upgrade might clarify all three.

**Falsifiable prediction for v0.2.** If we upgrade retrieval (bge-large + reranker), retrieval_recall_at_5 should rise from 17% to at least 40% on a fixed 500-entry KB. If that holds, knowledge_similarity AUROC should rise from 0.484 to at least 0.60 on the same eval set. If THAT holds, retrieval-injection-into-answer should swing positive (+0.05 or more delta). These three predictions are a tight test of the pipeline-is-the-bottleneck hypothesis.


## P16. Strategic decision: retrieval upgrade + finish v0.1 + v0.2 for EMNLP 2026

**Decided on 2026-04-24 after cross-model diagnostic studies (P14, P15) and honest workshop-paper feasibility assessment.**

**Plan.** Option B + C in the options I laid out. Concretely:

**Month 1 (now) — retrieval upgrade + v0.1 completion.**
- Upgrade retrieval pipeline: swap `nomic-embed-text` → `bge-large-en-v1.5`, add `bge-reranker-base` for top-20 → rerank to 5. Measure retrieval_recall_at_5 before/after on a 60-query probe set (should rise from 17% to ≥40% if the hypothesis in P15 is right).
- Re-run knowledge_similarity + answer_quality diagnostics on qwen2.5:7b with improved retrieval. Confirm three predictions from P15:
    1. retrieval_recall_at_5 rises to ≥40%
    2. knowledge_similarity AUROC rises from 0.484 toward ≥0.60
    3. answer_quality delta swings positive
- Llama GSA replication at different seed (still pending) to confirm Llama-is-dead isn't a fluke.
- Main experiment on all 3 local models at n=1000 with upgraded retrieval. ~$165 cloud budget.
- Write up v0.1 results as a technical report in the repo (`results/experiment/v0.1_report.md`). Not submitted as a paper; this is the foundation for v0.2.

**Months 2-3 — v0.2 design and execution.**
- Design document for v0.2: the non-stationary multi-armed bandit framing. Specifies the 5 experiments from LAB_NOTES P13 (capability-growth curve, per-signal AUROC evolution, threshold adaptation, non-stationary Thompson, escalation-feedback-loop robustness).
- Implement non-stationary Thompson sampling (exponential discounting + sliding window variants).
- Run the sequential growing-memory experiment on at least 2 datasets (MMLU-Pro + TriviaQA or BBH) × at least 2 local models.
- Estimated cloud budget: $250-400.

**Months 4-5 — paper writeup for EMNLP 2026 main track.**
- Target: "LLM Routing with Knowledge-Growing Arms: A Non-Stationary Bandit Framing"
- Deadline: ~end of October 2026 (if EMNLP 2026 follows EMNLP 2025's pattern).
- Primary contribution: formalizing LLM routing as a non-stationary bandit, empirical demonstration that classical NS-bandit machinery improves on stationary baselines in this regime.
- Secondary contribution: the cross-model ablation from v0.1.

**Month 6 — product launch.**
- Polish the v0.1 framework code.
- Blog post + HN + r/LocalLLaMA timed with paper public release.
- Establish whether there's user interest for v0.3+ work.

**Why this plan and not alternatives.**

- **Not a workshop-only paper on cross-model ablation:** the cross-model finding alone is interesting but narrow. It's a stepping stone, not the main story. Submitting it as the primary work would shortcut the actually-novel framing (NS-bandit + growing arms) we've only recently understood.
- **Not product-only:** research rigor buys trust; product without rigor fails against incumbents.
- **Not rushed workshop submission:** a weak workshop paper can hurt credibility more than no paper.

**Total investment estimate.**
- Cloud: ~$400-600 across v0.1 + v0.2 + retrieval experiments.
- Human time: ~150-200 hours over 6 months, feasible part-time.

**Key risks.**
1. Retrieval upgrade doesn't move the numbers. If `bge-large` + reranker doesn't bring retrieval_recall to ≥40%, we have deeper pipeline issues and v0.2's growing-memory story gets harder to tell.
2. Non-stationary Thompson doesn't beat stationary Thompson empirically. If the growing-memory dynamic is too slow to matter at reasonable N, the paper's central claim weakens.
3. Someone publishes the NS-bandit + LLM routing framing before us. Fast-moving field. Mitigation: post an arXiv preprint as soon as v0.2 has a defensible result.
4. Mistral's finding that "retrieval slightly hurts" turns into a real effect at larger n. Would complicate the product story. Mitigation: the product stays opt-in per-model.

**Success criteria for each stage.**

- Retrieval upgrade success: retrieval_recall_at_5 ≥ 40% AND knowledge_similarity AUROC ≥ 0.60 AND retrieval-injection delta > +0.05 with p < 0.1 at n=100.
- v0.1 main experiment success: at least one signal combo beats routellm_plus_ks by ≥ 0.05 AUROC with non-overlapping CIs on ≥ 2 of 3 models.
- v0.2 success: NS-Thompson beats Static-Thompson by ≥ 5% accuracy at fixed cost budget on the sequential experiment, with p < 0.05.
- Paper success: accept at EMNLP 2026 main or an equivalent venue (ICLR, NAACL, ACL).
- Product success: ≥ 500 GitHub stars in first 30 days after launch; ≥ 50 issues/PRs in the first 3 months; at least one serious company deployment (even if a friend-of-friend).

**Next immediate step.** Llama GSA replication + retrieval upgrade design.


## P17. Llama replication confirms: no self-assessment signal at 7-8B scale

**Follow-up to P14.** Replicated the GSA prompt study on `llama3.1:8b` at eval_seed=43 (disjoint test queries from the original seed=42 run), same n=100, same no-KB setting.

**Results — two independent seeds:**

| Seed | current | direct | confidence | prediction | Max AUROC | "Winner" |
|------|---------|--------|------------|------------|-----------|----------|
| 42 | 0.545 | 0.535 | 0.519 | 0.515 | 0.545 | `current` |
| 43 | 0.532 | 0.530 | 0.424 | 0.506 | 0.532 | `current` |

Both seeds: max AUROC ≤ 0.55. All four variants within 0.03 AUROC of each other on seed 42, within 0.11 on seed 43 (driven by `confidence` inverting to 0.424). Given bootstrap CI half-width ≈ ±0.10 at n=100, nothing here is statistically distinguishable from chance.

**Cross-model comparison:**

| Model | Best AUROC across seeds | Signal status |
|---|---|---|
| qwen2.5:7b | 0.636 | Real signal (CI lower bound > 0.5) |
| llama3.1:8b | 0.545 | **Statistically indistinguishable from chance** |
| mistral:7b-instruct | 0.747 | Strong signal |

**Interpretation.** Llama-3.1-8B's self-assessment is not just weakly calibrated — it is effectively uncalibrated for MMLU-Pro questions at this scale. No prompt framing rescues it. This is a cleanly reproducible negative finding, and it contradicts any "confidence prompting works for all small LLMs" claim we might have shipped.

**Speculation on cause.** Qwen2.5 and Mistral both include self-critique / calibration-flavored RLHF data in their postraining (per public model cards). Llama-3.1-Instruct's RLHF program emphasizes helpfulness and harmlessness; calibration is not an explicit objective. Per Kadavath et al. 2022, calibrated self-assessment emerges from calibration-aware training. This is consistent with what we observe, but we can't prove causation from two data points per model.

**What this means for v0.1 product shipping.**

If the product ships with a default self-assessment signal using a fixed prompt:
- Qwen-2.5 users (confidence prompt): roughly 0.64 AUROC — modest but real signal.
- Mistral users (direct prompt): roughly 0.75 AUROC — strong signal.
- Llama-3.1 users (any prompt): effectively no signal. The self-assessment contributes noise to the Thompson fusion.

**The product decision.** Either (a) auto-probe on install to select the winning prompt per user's local model, failing gracefully to "self-assessment disabled" when no prompt beats chance, or (b) document in the README that self-assessment works on Qwen2.5 and Mistral but is unreliable on Llama-3.1.

I lean (a) for the product. The auto-probe is 10 minutes and $0.50, cheaper than having a user post a bad review about Llama routing.

**What this means for the v0.1 cross-model main experiment.** The main experiment (Task 14, n=1000) includes Llama. We should expect Llama's best-combo AUROC to underperform Qwen and Mistral specifically because the self-assessment signal is dead. The memo should frame this as "GSA signal fails on Llama, but logprob_uncertainty and knowledge_similarity may still carry on" — measurable, not a disaster.

**Falsifiable prediction the replication strengthens.** If the Kadavath calibration-training hypothesis is right, models optimized with self-critique / calibration-aware RLHF should show stronger GSA signals than models without. Future experiments with DeepSeek-R1 (explicit reasoning training), Claude-Haiku-via-Bedrock-as-local (if cost allows), or a base Llama-3.1 vs. instruct variant would test this. Not in v0.1 scope; good v0.2+ follow-up.


## P18. Retrieval upgrade unblocks Task 14 — summary

See `results/experiment/EXPERIMENT_LOG.md` → EXP-001 for the full record. High-level outcome: swapping `nomic-embed-text` for `qllama/bge-large-en-v1.5` moved retrieval_recall_at_5 from ~17% to 91% (threshold-free, in-category). Answer-quality delta went from +0.017 (p=1.0 at n=60) on nomic to +0.12 (p=0.012 at n=100) on bge-large. The P15 "retrieval is the bottleneck" hypothesis is confirmed for the qwen arm. 2/3 gate predictions passed; Task 14 unblocked.

**Side finding worth flagging.** bge-large's similarity-score distribution sits ~0.15 lower than nomic's on this KB. The 0.75 `AutodidactConfig.similarity_threshold` default — which was calibrated to nomic — now filters out most hits (10% pass-rate vs 91% for in-category recall measured without a threshold). For the main experiment, we need to either:
1. Lower the `similarity_threshold` default to ~0.60 (simple, pragmatic), or
2. Remove the threshold concept entirely and let consumers decide per-call (cleaner but more surgery), or
3. Compute per-embedder thresholds from KB self-similarity at install time (most robust, not needed for v0.1).

**Lesson:** similarity-score thresholds are embedder-specific and cannot be shared across embedder swaps. This should be documented wherever we expose thresholds as configuration.

**Second side finding.** GSA prompt ranking changes when retrieval is turned on. In P14 (no retrieval), qwen's best prompt was `confidence` at AUROC 0.636. In EXP-001 (retrieval on), `prediction` wins at 0.568, `confidence` drops to 0.567, `current` inverts. Per-model-per-retrieval-condition prompt selection is a v0.1.5 research thread.


## P19. Retrieval thresholds are per-consumer, not global

See `results/experiment/EXPERIMENT_LOG.md` → EXP-002 for the full record.

**Short version:** I hypothesized that dropping `AutodidactConfig.similarity_threshold` from 0.75 (nomic-era) to 0.60 (bge-large calibration) would lift `knowledge_similarity` AUROC and GSA AUROC. EXP-002 falsified both predictions on qwen2.5:7b / n=60.

- `knowledge_similarity` AUROC is flat at chance (0.45-0.52) for every threshold in {0.50, 0.55, 0.60, 0.65, 0.70, 0.75}. Bootstrap CIs at n=60 mean the 0.068 best-vs-worst gap is noise.
- GSA AUROC actually improves with HIGHER threshold: chance at 0.50-0.60, then ~0.58-0.62 at 0.65-0.75 for the `confidence` and `direct` variants.

**Mechanism (repeating P9):** marginal retrieved content (scores 0.50-0.65) confuses GSA. The model sees tangentially-related context, tries to answer from it, and gets calibrated-looking-but-wrong self-assessments. At high threshold, GSA either sees strong context or honest absence, both of which calibrate correctly.

**Decision:** stop trying to pick ONE global threshold for all retrieval consumers. Different consumers need different thresholds:
- GSA prompt: high threshold (0.70+) — strong hits only.
- Answer-injection in main prompt: medium threshold (~0.60) — EXP-001 showed this helps accuracy.
- `knowledge_similarity` as feature for ML: no threshold; use raw max_sim (this is Change B in Task 13.7.4).

New Task 13.7.8 in the spec adds a `min_similarity` parameter to `KnowledgeStore.search()` so each consumer sets its own. `AutodidactConfig.similarity_threshold = 0.75` stays as the global fallback.

**Lesson.** A single config value for a thing that consumers care about differently is an abstraction that costs more than it saves. When I find myself debating "what threshold is right," I should ask instead "right for whom?"
