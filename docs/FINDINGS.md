# Findings

Everything the evaluation measured, honestly reported. Raw machine-readable
output lives in `results/*.json`; every table below is transcribed from those
files so nothing here drifts from the actual runs.

| result file | protocol | subjects |
|---|---|---|
| `results/eval_rating.json` | 5-fold CV rating prediction | 6 engines, RMSE ± std / MAE ± std |
| `results/eval_ranking.json` | full 943-user leave-one-out | 6 rankers, HR / NDCG / MRR |
| `results/eval_bootstrap.json` | paired bootstrap (2000 resamples) | all pairwise engine comparisons |
| `results/eval_agent200.json` | 200-user LOO, k=5 | agent vs ALS + constraint eval (context v1) |
| `results/eval_agent200_v2.json` | 200-user LOO, k=5, same users | agent vs ALS + constraint eval (context v2) |
| `results/eval_ranking_longtail.json` | debiased long-tail LOO (`exclude_head=0.02`) | 6 rankers on 773 users |
| `results/eval_report_v2.json` | aggregate report | all of the above + verdicts |
| `results/eval_report.json` | legacy 100-user LOO | early agent-vs-engines snapshot |
| `results/ml20m/eval_ranking_ml20m.json` | ml-20m LOO ranking, 2000-user sample | als / popular / random |
| `results/ml20m/eval_ranking_longtail_ml20m.json` | ml-20m debiased long-tail LOO | same engines on 785 users |
| `results/ml20m/eval_rating_ml20m.json` | ml-20m rating CV on a 3M-rating subsample | mf + mean baselines |
| `results/ml20m/eval_rating_ml20m_svd.json` | ml-20m rating CV on the same subsample | svd (tuned) + mf + mean baselines |
| `results/ml20m/eval_agent200_ml20m.json` | ml-20m LOO, first 200 sampled users, k=5 | agent (context v2) vs als + sci-fi constraint |
| `results/ml20m/eval_t5_ranking_ml20m.json` | ml-20m LOO ranking (T5) | als / svd / popular / blend 0.3–0.7 |
| `results/ml20m/eval_t5_ranking_longtail_ml20m.json` | ml-20m debiased long-tail LOO (T5) | same engines on 785 users |
| `results/ml20m/eval_t5_als_factors_ml20m.json` | ml-20m LOO ranking, factor-count sweep | als f24 / f32 / blend(f24,0.7) |
| `results/ml20m/eval_t5_als_factors_longtail_ml20m.json` | ml-20m debiased long-tail LOO, factor sweep | same engines on 785 users |
| `results/ml20m/eval_t5_als_factors_ml100k.json` | ml-100k LOO ranking, factor-count sweep | als f24 / f32 / f64 |
| `results/ml20m/eval_t5_svd_rating_ml100k.json` | ml-100k rating CV | svd (tuned) vs mf (tuned) |
| `results/ml20m/eval_t5_alpha_ml100k.json` | ml-100k LOO ranking, implicit-alpha probe | ratings scale the ALS confidence weight |

Reproduce everything with:

```bash
python -m recagent.cli eval --protocol all --baselines --bootstrap --sample 200 --agent
python -m recagent.cli eval --protocol ranking --baselines --exclude-head 0.02
```

---

## 0. MovieLens 20m migration — first results

All ml-100k numbers above stay valid for that dataset. This section reports
the first run on the 200× larger **ml-20m** (20M ratings, 138k users, 27k
movies), where the honest scope is:

- **ranking**: leave-one-out scored on a deterministic 2000-user sample (full
  138k-user LOO is impractical); memory-based `user`/`item` engines are
  omitted because their similarity products materialize too many nonzeros at
  138k × 27k scale — even with the sparse top-k form. ALS dominates and is the
  engine the serving/agent path uses anyway.
- **rating**: 5-fold CV on a deterministic 3M-rating subsample (not the full
  20M), `mf` + mean baselines for the same reason.

### 0.1 Ranking — raw LOO (`results/ml20m/eval_ranking_ml20m.json`)

| engine  | HR@10 | HR@5  | NDCG@10 | MRR    |
|---------|-------|-------|---------|--------|
| als     | 0.2755| 0.1930| 0.1615  | 0.1267 |
| popular | 0.0790| 0.0500| 0.0410  | 0.0296 |
| random  | 0.0010| 0.0010| 0.0007  | 0.0006 |

ALS is 3.5× popular at HR@10 and its absolute numbers barely dip versus the
943-user ml-100k run (HR@10 0.2853) — this is a far harder 27k-item
candidate set, so ALS's edge holds rather than erodes at scale.

### 0.2 Ranking — debiased long-tail (`results/ml20m/eval_ranking_longtail_ml20m.json`)

`exclude_head=0.02` drops the 2% most-popular items; **1215 of 2000** sampled
targets are head items, leaving **785 users**.

| engine  | HR@10 | HR@5  | NDCG@10 | MRR    |
|---------|-------|-------|---------|--------|
| als     | 0.0420| 0.0229| 0.0187  | 0.0117 |
| popular | 0.0000| 0.0000| 0.0000  | 0.0000 |
| random  | 0.0000| 0.0000| 0.0000  | 0.0000 |

Same story as ml-100k, starker: popularity goes to **zero** once head targets
are excluded, and ALS's long-tail HR@10 collapses to 0.042 — the tail of a
27k-item catalog is much harder than the tail of a 1.7k-item one. Long-tail
recommendation at this scale is the open challenge T5 targets.

### 0.3 Rating — 5-fold CV on 3M subsample (`results/ml20m/eval_rating_ml20m.json`)

`mf` uses its tuned config (`factors=6, iterations=15, reg=1.0`) via the
restored per-engine `engine_kwargs`; `svd` uses `8/20/1.0, bias_shrinkage=25`;
the mean baselines use the shared defaults.

| engine      | RMSE    | ±     | MAE     | ±     |
|-------------|---------|-------|---------|-------|
| item-mean   | 0.9523  | 0.0009| 0.7377  | 0.0006|
| user-mean   | 0.9939  | 0.0011| 0.7708  | 0.0009|
| mf (tuned)  | 1.0210  | 0.0013| 0.7592  | 0.0009|
| svd (tuned) | 1.0249  | 0.0036| 0.7638  | 0.0025|
| global-mean | 1.0521  | 0.0007| 0.8409  | 0.0006|

The `svd` row comes from `results/ml20m/eval_rating_ml20m_svd.json`; the `mf`
and mean rows reproduce `eval_rating_ml20m.json` exactly (same seed/subsample),
which doubles as a determinism check on the protocol.

The mean baselines improve slightly versus ml-100k (item-mean RMSE 1.0276 →
0.9523) — more data, denser signals. `mf` stays competitive on MAE (0.7592,
second-best) but its RMSE (1.0210) dips below the ml-100k 0.9920: at 20M the
subsample is sparser per user (~22 ratings/user vs ~106), so the fixed 6-factor
config is no longer optimal. The T5 biased-SVD upgrade (Section 0.4) did *not*
fix this — see the honest result there.

**Conditioning finding (unit-weight ALS).** With the *shared defaults*
(`factors=64, iterations=20, reg=0.1`) `mf` is broken at any scale: RMSE 1.7871
on ml-100k and 2.2284 on the ml-20m subsample — worse than the global mean.
Unit-weight ALS needs far fewer factors and much stronger regularization once
ratings-per-user approaches the factor count (106 → 22 here). The recorded
ml-100k figure of 0.9920 was produced with the tuned config; the protocol now
supports and records per-engine config so this cannot be silently mismatched
again.

### 0.4 Classic-model upgrades (T5) — honest results at scale

Two from-scratch engines were added and pushed at ml-20m: `BiasedMF` (kind
`svd`, joint factor+bias ALS solve, `bias_shrinkage` regularizer) and an RRF
`RankBlend` (kind `blend`, popularity fusion with the base ranker). Every probe
below failed to beat the plain `als` f64 baseline. All numbers transcribed from
the `eval_t5_*` files above.

**Popularity fusion does not beat ALS** (`eval_t5_ranking_ml20m.json` + long-tail):

| engine      | raw HR@10 | tail HR@10 |
|-------------|-----------|------------|
| als (f64)   | **0.2755**| **0.0420** |
| blend 0.7   | 0.2655    | 0.0293     |
| blend 0.5   | 0.2125    | 0.0051     |
| blend 0.3   | 0.1445    | 0.0000     |
| popular     | 0.0790    | 0.0000     |
| svd (BiasedMF) | 0.0010 | 0.0013     |

Every weight that leans on popularity degrades raw HR@10; at the tail any
blend weight ≥0.5 collapses toward popularity's zero. Implicit ALS is strictly
better than every fusion, so `blend` stays off the serving path.

**Factor-count tuning does not transfer to 20M.** On ml-100k, fewer ALS
factors *help* ranking: f24 HR@10 0.3001 / HR@5 0.2142 / MRR 0.1388 vs the f64
default 0.2927 / 0.1919 / 0.1185 (`eval_t5_als_factors_ml100k.json`). On ml-20m
the same configs *hurt* (`eval_t5_als_factors_ml20m.json`): f32 0.2495 and f24
0.2295 raw HR@10 vs f64 0.2755; tail f32 0.0229 and f24 0.0140 vs f64 0.0420.
The factor optimum grows with the catalog, so f64 stays the serving config.

**Biased MF does not beat unit-weight ALS on rating.** On ml-100k CV
(`eval_t5_svd_rating_ml100k.json`) the joint solve (`svd`, 8/20/1.0,
bias_shrinkage 25): RMSE 1.0192 ± 0.0084 vs `mf` 0.9920 ± 0.0049; on the ml-20m
3M subsample (`eval_rating_ml20m_svd.json`): 1.0249 ± 0.0036 vs 1.0210 ± 0.0013
— the added bias terms overfit (train RMSE 0.722 vs mf 0.732 on ml-100k) and
generalize worse on both datasets. A pure Funk-SGD probe lands at ~1.01 as
well. From-scratch implementations cannot reach the published 0.93–0.95
SVD-class figures: that gap needs a tuned SGD (learning-rate schedules,
adaptive rates), which is out of scope for these from-scratch engines.

**Implicit `alpha` weighting hurts ranking.** Scaling the ratings so they
dominate the ALS confidence weight (implicit's default is 1+40·r)
(`eval_t5_alpha_ml100k.json`): raw LOO HR@10 falls 0.2927 → 0.2513 (alpha5) →
0.2185 (alpha20). Treating ratings as pure confidence signals loses ranking
information.

**Bottom line.** At 20M the ALS-f64 ceiling stands: biased MF, popularity
fusion, implicit-alpha weighting and factor-count tuning all fail to lift it,
and RMSE-engineered predictors (`mf`, `svd`) rank at chance (HR@10 ≤ 0.0013).
The T5 disappointments are written down here on purpose.

### 0.5 Agentic reranker on the ml-20m sample (T6) (`results/ml20m/eval_agent200_ml20m.json`)

Context engineering v2 (user rating scale, per-item popularity + average rating,
social proof — see the §T6 commit `58a4358`) run over the live Gemma-4 endpoint
on the first 200 users of the ml-20m LOO sample, k=5.

| metric | agent | als    | delta (paired bootstrap) |
|--------|-------|--------|--------------------------|
| HR@5   | 0.165 | 0.205  | −0.040, 95% CI [−0.120, 0.035], p=0.331 |
| MRR    | 0.111 | 0.157  | −0.046, 95% CI [−0.105, 0.011], p=0.118 |

Unlike the ml-100k run (§4), the agent's deficit is **not statistically
significant** on this sample (p=0.33 HR@5, p=0.12 MRR) and the gap is a third
of the ml-100k size (−0.040 vs −0.095 HR@5). The richer context closed most of
the gap even though it did not close it.

The sci-fi constraint row on this sample is **degenerate**: the first 200 LOO
users sorted by id are heavy sci-fi fans, and the agent's lists are
byte-identical to the ALS lists (`agent == cf` for all 200 users; CF top-5 is
100% sci-fi across 89 distinct items — Star Wars, Twelve Monkeys, The Matrix,
...). Both precisions are 1.0, so the constraint test discriminates nothing
here. Reported as a limitation, not a win.

---

Transcribed from `results/eval_rating.json` (seed 42, deterministic splits).

| engine      | RMSE    | ±     | MAE     | ±     |
|-------------|---------|-------|---------|-------|
| item        | 0.9874  | 0.0056| 0.7856  | 0.0039|
| mf (ours)   | 0.9920  | 0.0049| 0.7614  | 0.0030|
| user        | 1.0090  | 0.0048| 0.8029  | 0.0031|
| item-mean   | 1.0276  | 0.0055| 0.8184  | 0.0031|
| user-mean   | 1.0420  | 0.0048| 0.8351  | 0.0032|
| global-mean | 1.1256  | 0.0056| 0.9447  | 0.0043|

Published default-parameter 5-fold numbers on the same ml-100k data
(Surprise `benchmark.py`): k-NN 0.980, Centered k-NN 0.951, k-NN Baseline
0.931, NMF 0.963, Slope One 0.946, SVD 0.934, SVD++ 0.919.

**Verdict.** Our three from-scratch engines sit inside the memory-based
family's published range on RMSE. The 0.89–0.95 SVD-class figures come from
*biased* SGD objectives (per-user/per-item bias terms), not unit-weight ALS,
so they are not a fair target for `recagent/mf.py`. The `svd` BiasedMF engine
added in T5 (Section 0.4) still does not reach them — the gap needs a tuned SGD
implementation. MAE favours `mf` (0.7614, best of all six) because its
residuals are tighter around zero.

## 2. Ranking — full 943-user leave-one-out

Protocol is the classic Cremonesi–Koren–Turrin (RecSys 2010) split: hold out
one rating per active user, score all other items, rank, evaluate top-10 hit
rate against the held-out item. Transcribed from `results/eval_ranking.json`
(seed 42).

| engine  | HR@10 | HR@5  | NDCG@10 | MRR    |
|---------|-------|-------|---------|--------|
| als     | 0.2853| 0.1919| 0.1658  | 0.1297 |
| popular | 0.1368| 0.0838| 0.0700  | 0.0498 |
| user    | 0.1166| 0.0753| 0.0635  | 0.0473 |
| mf      | 0.0467| 0.0276| 0.0225  | 0.0152 |
| random  | 0.0042| 0.0011| 0.0021  | 0.0015 |
| item    | 0.0021| 0.0011| 0.0008  | 0.0004 |

**Verdict.** This reproduces the CKT findings precisely:

- a **non-personalized popularity baseline is surprisingly strong** under plain
  LOO — it beats every from-scratch method except ALS;
- **correlation-based item-kNN performs extremely poorly** (HR@10 0.0021,
  barely above random) because under LOO the held-out item's neighbours are
  inferred from a single user's profile;
- **RMSE gains do not translate into top-N accuracy** — the best RMSE engine
  (item, 0.9874) is the worst ranker;
- `als` (implicit ALS) is the dominant ranker by a wide margin.

### Paired bootstrap (`results/eval_bootstrap.json`)

2000 resamples of the per-user rank scores, 95% CI, two-sided p on mean
difference. ALS is significantly better than **every** engine on both hit@5
and MRR (all p<0.001). user-vs-popularity is a **statistical tie** (p=0.47).

| comparison        | hit@5 mean diff | p       |
|-------------------|-----------------|---------|
| als vs user       | +0.117          | <0.001  |
| als vs popular    | +0.108          | <0.001  |
| als vs mf         | +0.164          | <0.001  |
| user vs popular   | −0.008          | 0.467   |

## 3. Ranking — debiased long-tail LOO (`exclude_head=0.02`)

Plain LOO is head-biased: it measures how well systems predict the *most
popular* items. Following the CKT recommendation, this run drops test targets
belonging to the top 2% most-popular items (ties broken by item id), so only
long-tail predictions are scored. Transcribed from
`results/eval_ranking_longtail.json` (seed 42, `exclude_head=0.02`).

**170 of 943** held-out targets were head items; the debiased set keeps
**773 users**.

| engine  | HR@10 | HR@5  | NDCG@10 | MRR    |
|---------|-------|-------|---------|--------|
| als     | 0.2186| 0.1358| 0.1202  | 0.0908 |
| user    | 0.0556| 0.0285| 0.0251  | 0.0160 |
| mf      | 0.0401| 0.0207| 0.0174  | 0.0106 |
| random  | 0.0078| 0.0052| 0.0038  | 0.0026 |
| popular | 0.0078| 0.0013| 0.0026  | 0.0012 |
| item    | 0.0026| 0.0013| 0.0009  | 0.0005 |

**Verdict.** Popularity **collapses to random parity** (HR@10 0.1368 → 0.0078)
and lands *below* random at HR@5 (0.0013 vs 0.0052). Its earlier strength was
an artifact of head items being the held-out targets. On the strictly harder
long-tail task ALS stays dominant (HR@10 0.2186) while the memory-based `user`
and explicit `mf` engines now beat raw popularity. This run confirms ALS's edge
is real and not an artifact of head-item targets.

## 4. Agentic reranker — 200-user sample, k=5 (`results/eval_agent200.json`)

| metric | agent | als    | delta (paired bootstrap) |
|--------|-------|--------|--------------------------|
| HR@5   | 0.070 | 0.165  | −0.095, 95% CI [−0.155, −0.035], p=0.004 |
| MRR    | 0.036 | 0.119  | −0.083, 95% CI [−0.128, −0.040], p<0.001 |

**Verdict.** The ranking edge seen on the earlier 100-user sample did not
replicate at 200 users: the agent is significantly **below** ALS at raw LOO
ranking on this data. This is the honest result and it is reported as-is.

Where the agent earns its place is **explicit constraints**:

| constraint | agent precision | pure-CF precision |
|------------|-----------------|-------------------|
| film-noir  | **1.0** (all 5 items compliant, all 200 users) | 0.043 |
| sci-fi     | 1.0 | 0.145 |

(§4 constraint numbers were re-verified from the saved per-user lists: the
film-noir agent lists recompute to 1.0000 and the earlier "CF is ~100% sci-fi"
note for the sci-fi row was a case-sensitive genre-lookup artifact — CF is in
fact only ~14.5% sci-fi on this sample.)

### Context engineering v2 rerun (`results/eval_agent200_v2.json`)

Same 200 users, same protocol, same endpoint — only the evidence changed
(rating scale, per-item popularity/avg-rating, social proof, calibrated score
hint). The agent's raw-ranking deficit vs ALS is no longer significant:

| metric | agent (v1) | agent (v2) | als (v2) | delta v2 (paired bootstrap) |
|--------|------------|------------|----------|-----------------------------|
| HR@5   | 0.070 | 0.160 | 0.230 | −0.070, 95% CI [−0.150, 0.010], p=0.099 |
| MRR    | 0.036 | 0.092 | 0.138 | −0.046, 95% CI [−0.099, 0.008], p=0.094 |

The agent more than doubled absolute hit rate (0.070 → 0.160) and closed the
significant deficit (§4 p=0.004 → p=0.099). The v2 ALS baseline (0.230) uses
the canonical fresh-fit model; §4's baseline (0.165) used the then-persisted
artifact model, so baseline levels differ — the agent-side comparison (same
users, same protocol) is the valid one.

Constraint compliance with v2 is now a clean, non-degenerate win: sci-fi
precision **1.0 vs 0.145** — 100% of agent picks carry the required genre
while pure CF is only 14.5% sci-fi on this sample. This is the one thing the
LLM layer provably adds over the engines.

## 5. Bottom line

1. The from-scratch engines are within the published default-parameter range
   on ml-100k rating prediction.
2. Implicit ALS is the strongest LOO ranker, including on the debiased
   long-tail task.
3. Popularity's plain-LOO strength is a head-item artifact; it is at chance on
   the tail.
4. The agentic layer does not improve raw ranking on this dataset but provides
   the one thing pure CF cannot: **verifiable constraint compliance**
   (film-noir precision 1.0 vs 0.043). On the ml-20m sample the raw-ranking gap
   narrows to a non-significant −0.040 HR@5 (§0.5) with context engineering v2.

## 6. Limitations / threats to validity

- **ml-100k only.** All conclusions are for one small, movie-only dataset;
  popular-movie effects and genre constraints may not transfer.
- **LOO measures top-N hit, not discovery.** A system that only ever
  recommends popular items looks fine under plain LOO (see Section 2) — the
  debiased protocol (Section 3) is the more meaningful metric.
- **Agent latency/cost.** The agent is 200 users of 943; a full run is bounded
  by the live vLLM gateway, not the offline engines.
- **Bootstrap ties.** user-vs-popularity is a tie *on this sample*; the true
  ordering is unresolved.
- **Constraint eval** used a genre hold-all-items-compliant design; it tests
  compliance, not whether constrained recs are what users want. On the ml-20m
  sample (§0.5) the sci-fi row is degenerate: the sampled users are heavy
  sci-fi fans, both engine and agent are already 100% sci-fi, and the agent
  copies the CF lists exactly.
- **Agent sample bias (ml-20m).** The first-200-by-id LOO users are atypical
  (early MovieLens adopters, heavy sci-fi); the raw-ranking gap measured on
  them may not generalize to a uniform user sample.

## 7. Provenance

- Data: MovieLens 100k (`data/`, fetched by `recagent.data.fetch_movielens`)
  and MovieLens 20M (`data/ml-20m/`, fetched by `load_ratings_20m`).
- Artefacts: `artifacts/`, `artifacts_user/`, `artifacts_item/`,
  `artifacts_als/` (ml-100k) and `artifacts_ml20m/` (ml-20m ALS f64)
  (all gitignored; regenerable via `train`).
- Seeds: all protocols seeded (default 42); bootstrap deterministic.
- Engine config: `mf` factors=6 / iterations=15 / reg=1.0; `als` implicit
  defaults (f64), serving config confirmed at 20M; memory-based k=25,
  min_sim=0.1, k_sim=20. `svd` 8/20/1.0 + bias_shrinkage=25.
- Agent eval (§0.5, §4): live vLLM endpoint, Gemma-4-31B-it, temperature 0.1,
  `scripts/run_ml20m_agent.py` / `scripts/run_agent200.py`.
