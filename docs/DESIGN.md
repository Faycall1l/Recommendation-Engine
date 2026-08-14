# Design notes

How the system is built and *why* each decision was made. Findings live in
[FINDINGS.md](FINDINGS.md); test coverage in [TESTING.md](TESTING.md);
runbook in [OPERATIONS.md](OPERATIONS.md).

## Architecture at a glance

```
┌──────────────────────────────────────────────────────────────────┐
│  CLI (recagent.cli)   REST gateway (recagent.api)   SDK (recagent_sdk)
└───────────────┬───────────────────────────────────────────────────┘
                │
        RecClient (recagent.client)
          ├── candidate engines  recagent.engines / cf / mf
          ├── agentic reranker   recagent.agent + tools
          └── explainer          recagent.explain
                │
        eval harness  recagent.evaluate + scripts/run_*.py → results/
```

`RecClient` is the single facade: it owns the trained artefact (`State`), the
engine, the optional LLM agent, and the explainer. Everything a user can do —
CLI, REST, SDK — goes through it, so behaviour is identical in all three
frontends.

## Engines

Three candidate engines share one interface (`fit` / `recommend(matrix,
user_idx, n)`), built by `recagent.engines.build_engine(kind)`.

| kind    | implementation | notes |
|---------|----------------|-------|
| `als`   | `implicit.ALS` | weighted matrix factorisation; the classic baseline |
| `user`  | `recagent.cf.UserBasedCF` | memory-based, numpy only |
| `item`  | `recagent.cf.ItemBasedCF` | memory-based, numpy only |
| `mf`    | `recagent.mf.ExplicitALS` | from-scratch unit-weight ALS (explicit 1–5) |

### Memory-based methods (`recagent/cf.py`)

Ratings are **mean-centred** (subtract the baseline from the rating) before
similarity; predictions re-add the baseline. Negatives are floored to 0 so only
positively correlated neighbours vote.

**User-based (Pearson).** User similarity is cosine on centred rows, which
equals Pearson correlation restricted to co-rated items:

```
sim(u, v) = Σᵢ (r_ui − μ_u)(r_vi − μ_v) / (‖r_u − μ_u‖ · ‖r_v − μ_v‖)
r̂_ui = μ_u + Σᵥ sim(u,v)·(r_vi − μ_v) / Σᵥ |sim(u,v)|
```

`predict` is a single sparse row-normalised matmul; `score_all` batches the
same math into a dense matrix for eval. `k=25` neighbours, `min_sim=0.1`,
`k_sim=20` most-similar stored per user.

**Item-based (adjusted cosine).** Columns are mean-centred per item, then
`S = ĈᵀĈ` on the column-normalised matrix — the textbook adjusted-cosine
formula:

```
sim(i, j) = Σᵤ (r_ui − μ_i)(r_uj − μ_j) / (‖r_i − μ_i‖ · ‖r_j − μ_j‖)
r̂_ui = Σ_{j∈rated(u)} sim(i,j)·r_uj / Σ_{j∈rated(u)} |sim(i,j)|
```

Prediction uses only the user's own ratings as evidence, falling back to the
user's mean when no similar item is in the profile.

### Matrix factorisation (`recagent/mf.py`)

`ExplicitALS` is **unit-weight explicit ALS** (SGD-free, closed-form
alternation) with `factors=6, iterations=15, regularization=1.0`. It is tuned
for rating RMSE on ml-100k, *not* for ranking — see the decision log.

## Agentic reranker (`recagent/agent.py`)

`recagent chat` / `RecClient.recommend` run a **plan-then-execute** agent
(Gemma-4 via pydantic-ai) instead of in-loop tool calling:

1. `build_plan` — parse `user_id`, `k`, and any hard constraints (e.g. genre).
2. `build_evidence` — deterministic `ToolRegistry` calls: user profile,
   CF candidates, item metadata, and — for cold-start users or rare genres —
   popularity priors and `similar_items` chains.
3. one structured-output LLM call emits a ranked `RankedItems` list.
4. `_clean_items` guardrails drop hallucinations, constraint violations and
   duplicates, capping the list at `k`.

Why: vLLM's in-loop tool template is flaky (repeated identical calls, empty
parts). Plan-then-execute keeps the reasoning signal without the loop.

LLM config lives in `.env` (`ATHAR_AGENT__*`, see `.env.example`); without it
`RecClient` degrades to pure CF.

## Explainable recommendations (`recagent/explain.py`)

Every recommendation gets a verifiable "why", derived **deterministically from
the trained artefacts** — no LLM in the evidence path:

- **genre affinity** — the item's genres against the user's rating-weighted
  dominant genres;
- **engine score** and its **boost** over the user's mean rating;
- **similar-taste** — items the user already rated that are item-item similar
  to the recommendation;
- **taste overlap** — rated neighbours that also rated the item;
- popularity and quality fallbacks for cold-start users.

`explain_recommendation` classifies each case into one basis, in priority
order: `genre-affinity > similar-taste > taste-overlap > popularity`. The
`Explanation` model carries the deterministic fields (score, boost, matched
genres, similar rated items, snippet) plus an optional `llm` prose field.

`RecExplainer` turns the evidence into fluent prose in a **single
structured-output call** with an explicit no-invention rule: it must only
restate the supplied facts. An empty/failed LLM response falls back to the
deterministic `snippet`, so an explanation always exists.

## Evaluation harness (`recagent/evaluate.py`)

Honest-eval rules the whole project is built around:

1. **No data leakage.** LOO holds out one rating per user; CV splits partition
   all triples; the held-out item is excluded from the recommendation set.
2. **Baselines always.** Every ranking table ships with `popular`, `random`,
   and mean baselines so a "good" number is always relative to chance.
3. **Significance, not anecdotes.** Every pairwise comparison ships with a
   paired bootstrap (2000 resamples, 95% CI, two-sided p).
4. **Report the result, not the hope.** When the agent lost to ALS at 200
   users, the loss was reported.

Protocols:

- `cv_rating_eval_from_arrays` — 5-fold CV RMSE/MAE with std across folds.
- `loo_ranking_eval_from_arrays` — Cremonesi–Koren–Turrin LOO: hold out one
  rating per active user, score all other items, rank, top-10 hit rate.
  `exclude_head=f` debiases by dropping test targets in the top-fraction
  most-popular items (`head_item_ids`), the recommended fix for head bias.
- `paired_bootstrap` — resamples per-user scores, returns CI + p.
- `genre_precision` — for constraint eval: fraction of returned items that
  satisfy a constraint, computed for agent and pure-CF lists.

## Decision log

| decision | why | evidence |
|---|---|---|
| explicit unit-weight ALS for `mf` | SGD-fair comparison; matches "ALS" in the CF literature | RMSE 0.9920 (Section 1) |
| keep `als` from `implicit` | the strongest LOO ranker; weighting captures implicit signal | HR@10 0.2853 raw, 0.2186 long-tail |
| item-based collapse under LOO is *expected* | neighbours of a held-out item come from one user's profile | HR@10 0.0021 |
| popularity baseline in every table | RMSE≠ranking; plain LOO flatters popularity | HR@10 0.1368 raw vs 0.0078 tail |
| `exclude_head` debias | plain LOO measures head-item prediction, not discovery | 170/943 targets were head; popularity → random |
| plan-then-execute, not in-loop tools | vLLM in-loop tool template flaky | agent design |
| agent for constraints, not raw ranking | it loses at LOO but hits 1.0 film-noir compliance | agent200 |
| deterministic evidence, LLM prose last | explanations must be grounded even when the LLM is off | explain design |

## API / CLI / SDK surface

| capability | CLI | REST | SDK |
|---|---|---|---|
| recommend | `recagent recommend USER [--k] [--genre]` | `GET/POST /recommend` | `client.recommend` |
| explain | `recagent explain USER ITEM` | `POST /explain` | `client.explain` |
| chat | `recagent chat` | `POST /chat` | `client.chat` |
| feedback | — | `POST /feedback` | `client.feedback` |
| catalog / item info | `recagent item ID` | `GET /catalog`, `/items/{id}` | `client.item_info` |
| train | `recagent train [--cf]` | — | — |
| eval | `recagent eval [--protocol] [--baselines] [--bootstrap] [--agent] [--exclude-head]` | — | — |
| serve | `recagent serve --port` | — | — |

## Engineering conventions

- **Type hints everywhere; pydantic for every boundary** (requests, evidence,
  explanations, agent output) so malformed data fails loudly.
- **Numpy/sparse math only** for the engines; no sklearn/surprise in the hot
  path. LLM calls are the only network dependency.
- **Deterministic by default** — every split and eval path takes a `seed`;
  randomness exists only where the protocol demands it (bootstrap resampling).
- **Good habits the tests enforce:** no network at import time, no LLM call in
  CI (`.env` is gitignored), CLI tests never touch a live model, everything
  round-trips through `save`/`load` and `to_dict`.
- Every doc in `docs/` is regenerated-from-truth: numbers come out of
  `results/*.json`, never retyped from memory.
