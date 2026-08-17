<p align="center">
  <img src="assets/recagent-logo.svg" alt="RecAgent mascot" width="180"/>
</p>

# RecAgent

Structured LLM reasoning over collaborative filtering evidence. A plan-execute-reflect agent on Gemma-4/vLLM that produces ranked recommendations with verifiable traces, contrastive explanations, and hard constraint support.

> **Docs:** [Findings & evaluation results](docs/FINDINGS.md) · [Design notes](docs/DESIGN.md) · [Testing](docs/TESTING.md) · [Operations/runbook](docs/OPERATIONS.md)

## Engines

Three candidate engines share one interface (`fit` / `recommend(matrix, user_idx, n)`):

| kind    | implementation | notes |
|---------|----------------|-------|
| `als`   | `implicit.ALS` | weighted matrix factorisation, the classic baseline |
| `user`  | `recagent.cf.UserBasedCF` | memory-based, implemented from scratch in numpy |
| `item`  | `recagent.cf.ItemBasedCF` | memory-based, implemented from scratch in numpy |

The default engine is **user-based**. Choose with `--cf` on `train`.

### Memory-based methods (`recagent/cf.py`)

Both are classic neighbourhood methods over the explicit 1–5 rating matrix,
implemented from scratch (no sklearn/surprise).

**User-based (Pearson correlation)** — ratings are mean-centred per user
(`r_ui - μ_u`); user similarity is cosine on the centred rows, which equals
Pearson correlation restricted to co-rated items:

```
sim(u, v) = Σ_i (r_ui - μ_u)(r_vi - μ_v) / (‖r_u - μ_u‖ · ‖r_v - μ_v‖)
r̂_ui = μ_u + Σ_v sim(u, v)·(r_vi - μ_v) / Σ_v |sim(u, v)|
```

Negatives are floored to 0 (only positively correlated users vote). A single
sparse row-normalised matmul scores every item; `score_all()` batches the same
math into a dense `(n_users × n_items)` matrix for eval.

**Item-based (adjusted cosine)** — columns are mean-centred per item, then
`S = ĈᵀĈ` on the column-normalised matrix:

```
sim(i, j) = Σ_u (r_ui - μ_i)(r_uj - μ_j) / (‖r_i - μ_i‖ · ‖r_j - μ_j‖)
r̂_ui = Σ_j∈rated(u) sim(i, j)·r_uj / Σ_j∈rated(u) |sim(i, j)|
```

Prediction uses only the user's own ratings as evidence, falling back to their
mean when no similar item is in the profile.

## Agentic reranker

`recagent chat` / `RecClient.recommend` run a **plan-then-execute** agent
(Gemma-4 via pydantic-ai) instead of in-loop tool calling:

1. `build_plan` — parse `user_id`, `k`, and any hard constraints (e.g. genre).
2. `build_evidence` — deterministic `ToolRegistry` calls: user profile,
   collaborative-filtering candidates, item metadata, and — for cold-start
   users or rare genres — popularity priors and `similar_items` chains.
3. one structured-output LLM call emits a ranked `RankedItems` list.
4. **reflection loop** (opt-in `reflect=True`) — deterministic post-ranking
   checks (too few items, constraint violations, evidence coverage, genre
   diversity); a second targeted LLM call fixes issues when found.
5. **MMR diversity re-ranking** (`diversity=True`, `lambda_param=0.5`) — Jaccard
   genre similarity prevents the top-k from being all-one-genre.
6. `_clean_items` guardrails drop hallucinations, constraint violations and
   duplicates, capping the list at `k`.

This sidesteps vLLM's flaky in-loop tool template (repeated identical calls,
empty parts) while keeping the reasoning signal.

### Client resilience

`RecClient` retries transient vLLM failures (HTTP 502/503/504, connection
errors) with exponential backoff and jitter. A built-in circuit breaker opens
after 5 consecutive failures, preventing cascading timeouts when the endpoint
is down.

## Quickstart

```bash
python -m pip install -e .
python -m recagent.cli train --cf user           # user-based (default)
python -m recagent.cli recommend 196             # engine + top-n
python -m recagent.cli explain 196 64            # why an item for a user
python -m recagent.cli train --cf als            # switch engines
python -m recagent.cli eval --all-cf --sample 100 --agent
python -m recagent.cli serve --port 8000         # REST gateway (docs at /docs)
```

LLM config lives in `.env` (`ATHAR_AGENT__*`, see `.env.example`); without it
`RecClient` degrades to pure CF.

## SDK

```python
from recagent_sdk import RecommendClient

with RecommendClient("http://127.0.0.1:8000") as client:
    print(client.recommend(user_id=196, k=5, filters={"genre": "Sci-Fi"}))
```

## Results (100-user LOO holdout, `results/eval_report.json`)

| engine  | HR@1 | HR@3 | HR@5 | HR@10 |
|---------|------|------|------|-------|
| als     | 0.04 | 0.09 | 0.12 | 0.24 |
| user    | 0.03 | 0.11 | 0.12 | 0.16 |
| item    | 0.00 | 0.00 | 0.00 | 0.01 |
| agent   | 0.04 | 0.09 | 0.15 | —     |

Honest read: ALS remains the strongest raw LOO predictor; the from-scratch
user-based method is competitive at HR@3/5; item-based collapses because the
held-out item is the only evidence for its neighbours. The agentic reranker
matches ALS at HR@1/3 and edges ahead at HR@5 — and is the only path that can
honour hard constraints (verified 5/5 genre compliance live).

## Explainable recommendations

Every recommendation comes with a verifiable "why". `recagent/explain.py`
derives the evidence deterministically from the trained artefacts — no LLM in
the loop — so the explanation is always grounded:

- **genre affinity** — the item's genres against the user's rating-weighted
  dominant genres;
- **engine score** and its **boost** over the user's mean rating;
- **similar-taste** — items the user already rated that are item-item similar
  to the recommendation;
- **contrastive comparison** — when `recommended_ids` is provided, the
  explanation includes "why this over the next-best alternative?" (the user's
  highest-rated genre-sharing item that was NOT recommended);
- popularity and quality fallbacks for cold-start users.

```
python -m recagent.cli explain 196 64        # CLI (LLM prose when enabled)
POST /explain  {"user_id": 196, "item_id": 64}   # REST
client.explain(user_id=196, item_id=64)          # SDK
client.explain(user_id=196, item_id=64, recommended_ids={64, 12, 77})  # contrastive
```

The LLM restates the evidence as fluent prose in a single structured-output
call, held to the facts by an explicit no-invention rule; an empty response
falls back to the deterministic one-liner, so an explanation always exists.
Live example (user 196, Shawshank Redemption):

> You might enjoy The Shawshank Redemption (1994), as it is a Drama and you
> have previously liked Stand by Me (1986).
> — evidence: liked Secrets & Lies (5.0), English Patient (5.0), Stand by Me (5.0)

## State-of-the-art evaluation (`results/eval_report_v2.json`)

Three offline protocols, all reproducible from `recagent` (no external
recommender library), on ml-100k:

```
python -m recagent.cli eval --protocol all --baselines --bootstrap --sample 200
```

### Rating prediction — 5-fold CV RMSE/MAE (`results/eval_rating.json`)

| engine      | RMSE    | MAE     |
|-------------|---------|---------|
| item        | 0.9874  | 0.7856  |
| mf (ours)   | 0.9920  | 0.7614  |
| user        | 1.0090  | 0.8029  |
| item-mean   | 1.0276  | 0.8184  |
| user-mean   | 1.0420  | 0.8351  |
| global-mean | 1.1256  | 0.9447  |

`mf` is a from-scratch unit-weight explicit ALS (`recagent/mf.py`), tuned to
factors=6 / iterations=15 / reg=1.0. Published default-parameter 5-fold numbers
on the same data (Surprise `benchmark.py`): k-NN 0.980, Centered k-NN 0.951,
k-NN Baseline 0.931, NMF 0.963, Slope One 0.946, SVD 0.934, SVD++ 0.919. Our
engines sit inside the memory-based family's published range; the 0.89–0.95
SVD-class figures come from biased SGD objectives, not unit-weight ALS.

### Ranking — full 943-user leave-one-out (`results/eval_ranking.json`)

| engine  | HR@10 | HR@5  | NDCG@10 | MRR    |
|---------|-------|-------|---------|--------|
| als     | 0.2853| 0.1919| 0.1658  | 0.1297 |
| popular | 0.1368| 0.0838| 0.0700  | 0.0498 |
| user    | 0.1166| 0.0753| 0.0635  | 0.0473 |
| mf      | 0.0467| 0.0276| 0.0225  | 0.0152 |
| random  | 0.0042| 0.0011| 0.0021  | 0.0015 |
| item    | 0.0021| 0.0011| 0.0008  | 0.0004 |

This reproduces the classic Cremonesi–Koren–Turrin (RecSys 2010) findings:
a non-personalized popularity baseline is surprisingly strong under plain
leave-one-out, correlation-based item-kNN performs extremely poorly, and RMSE
gains do not translate into top-N accuracy. Paired bootstrap (2000 resamples,
`results/eval_bootstrap.json`): ALS is significantly better than every engine
on hit@5 and MRR (all p<0.001); user-vs-popularity is a statistical tie
(p=0.47).

The fix that finding calls for is implemented: **debiased long-tail LOO**
excludes the top-2% most-popular head items from the test set
(`--exclude-head`, `results/eval_ranking_longtail.json`). 170 of 943 held-out
targets were head items; on the remaining 773 users:

| engine  | HR@10 | HR@5  | NDCG@10 | MRR    |
|---------|-------|-------|---------|--------|
| als     | 0.2186| 0.1358| 0.1202  | 0.0908 |
| user    | 0.0556| 0.0285| 0.0251  | 0.0160 |
| mf      | 0.0401| 0.0207| 0.0174  | 0.0106 |
| popular | 0.0078| 0.0013| 0.0026  | 0.0012 |
| random  | 0.0078| 0.0052| 0.0038  | 0.0026 |
| item    | 0.0026| 0.0013| 0.0009  | 0.0005 |

Popularity collapses to random parity (and *below* random at HR@5), exposing
its earlier strength as an artifact of head items being the held-out targets.
ALS stays the strongest ranker on the strictly harder long-tail task; the
memory-based `user` and explicit `mf` engines now beat raw popularity.

### Agentic reranker — 200-user sample, k=5 (`results/eval_agent200_v2.json`)

| metric | agent (v1) | agent (v2) | als (v2) | delta v2 (paired bootstrap) |
|--------|------------|------------|----------|-----------------------------|
| HR@5   | 0.070 | 0.160 | 0.230 | −0.070, 95% CI [−0.115, −0.030], p=0.002 |
| MRR    | 0.036 | 0.092 | 0.138 | −0.046, 95% CI [−0.079, −0.016], p=0.005 |

Context engineering v2 more than doubles the agent's absolute hit rate
(0.070 → 0.160) but it remains significantly *below* ALS at raw LOO ranking.
Where it earns its place is explicit constraints — with the genre-lookup
artifact fixed, pure CF is only ~14.5% sci-fi on this sample, so the sci-fi
row is a clean win: agent precision **1.0 vs 0.145** (and film-noir 1.0 vs
0.043). The win reproduces on ml-20m (1.0 vs 0.179–0.184, lists never copies
of CF). Bootstrap p-values in this section are recomputed with per-user
aligned arrays; see §4 in [FINDINGS](docs/FINDINGS.md).

### MovieLens 20M migration (`results/ml20m/*`, FINDINGS §0)

Same protocol scaled to 20M ratings / 138k users / 27k items. Raw LOO on a
uniform 200-user sample: agent HR@5 **0.140 vs ALS 0.200, p=0.021**
(`eval_agent200_ml20m_uniform.json`) — the deficit is significant here too.
On the strict long tail (`exclude_head=0.2`) both engines score 0.000 at all
top-k, so the tail task is below top-k resolution at this catalog size.

### Bottom line

The from-scratch engines are within the published default-parameter range on
ml-100k rating prediction, and the implicit ALS is the strongest LOO ranker.
The agentic layer does not help raw ranking — the deficit vs ALS is
statistically significant in most cohorts (ml-100k v1/v2, ml-20m uniform) —
but it provides the one thing pure CF cannot: **verifiable constraint
compliance** (1.0 vs 0.043–0.184 across datasets, zero list-copying).
