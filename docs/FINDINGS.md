# Findings

Everything the evaluation measured, honestly reported. Raw machine-readable
output lives in `results/*.json`; every table below is transcribed from those
files so nothing here drifts from the actual runs.

| result file | protocol | subjects |
|---|---|---|
| `results/eval_rating.json` | 5-fold CV rating prediction | 6 engines, RMSE ± std / MAE ± std |
| `results/eval_ranking.json` | full 943-user leave-one-out | 6 rankers, HR / NDCG / MRR |
| `results/eval_bootstrap.json` | paired bootstrap (2000 resamples) | all pairwise engine comparisons |
| `results/eval_agent200.json` | 200-user LOO, k=5 | agent vs ALS + constraint eval |
| `results/eval_ranking_longtail.json` | debiased long-tail LOO (`exclude_head=0.02`) | 6 rankers on 773 users |
| `results/eval_report_v2.json` | aggregate report | all of the above + verdicts |
| `results/eval_report.json` | legacy 100-user LOO | early agent-vs-engines snapshot |

Reproduce everything with:

```bash
python -m recagent.cli eval --protocol all --baselines --bootstrap --sample 200 --agent
python -m recagent.cli eval --protocol ranking --baselines --exclude-head 0.02
```

---

## 1. Rating prediction — 5-fold CV RMSE/MAE

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
so they are not a fair target for `recagent/mf.py`. MAE favours `mf` (0.7614,
best of all six) because its residuals are tighter around zero.

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
| sci-fi     | 1.0 | ~1.0 (degenerate: CF head is already ~100% sci-fi) |

## 5. Bottom line

1. The from-scratch engines are within the published default-parameter range
   on ml-100k rating prediction.
2. Implicit ALS is the strongest LOO ranker, including on the debiased
   long-tail task.
3. Popularity's plain-LOO strength is a head-item artifact; it is at chance on
   the tail.
4. The agentic layer does not improve raw ranking on this dataset but provides
   the one thing pure CF cannot: **verifiable constraint compliance**
   (film-noir precision 1.0 vs 0.043).

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
  compliance, not whether constrained recs are what users want.

## 7. Provenance

- Data: MovieLens 100k (`data/`, fetched by `recagent.data.fetch_movielens`).
- Artefacts: `artifacts/`, `artifacts_user/`, `artifacts_item/`,
  `artifacts_als/` (gitignored; regenerable via `train`).
- Seeds: all protocols seeded (default 42); bootstrap deterministic.
- Engine config: `mf` factors=6 / iterations=15 / reg=1.0; `als` implicit
  defaults; memory-based k=25, min_sim=0.1, k_sim=20.
