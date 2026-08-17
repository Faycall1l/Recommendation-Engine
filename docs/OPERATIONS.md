# Operations / runbook

Commands for the day-to-day lifecycle. Design decisions in
[DESIGN.md](DESIGN.md), results in [FINDINGS.md](FINDINGS.md).

## Install

```bash
python -m pip install -e .        # package + recagent + recagent_sdk
cp .env.example .env              # optional LLM credentials (ATHAR_AGENT__*)
```

Without `.env`, `RecClient` runs pure CF; with it, `recommend`/`chat`/`explain`
use the LLM agent.

## Train

```bash
python -m recagent.cli train --cf user          # default: user-based -> artifacts/
python -m recagent.cli train --cf als           # -> artifacts_als/
python -m recagent.cli train --cf item          # -> artifacts_item/
python -m recagent.cli train --cf als --data-kind ml-20m   # big dataset -> artifacts_ml20m/
```

(`mf` is a from-scratch explicit ALS built at eval time via
`recagent.engines.build_engine("mf")`, not a `train` target.) Artefacts are
gitignored and regenerable; a saved artefact embeds the engine kind and its
fitted weights.

`--data-kind {ml-100k, ml-20m}` switches the dataset family everywhere
(`train`, `eval`). ml-20m is 20M ratings / 138k users / 27k movies — the
default ml-100k stays the fast, test-safe default.

## Recommend / explain

```bash
python -m recagent.cli recommend 196             # engine + top-10
python -m recagent.cli recommend 196 --k 5 --genre "Sci-Fi"   # constrained
python -m recagent.cli explain 196 64            # why this item for this user
```

Agent pipeline flags (on `RecAgent` / CLI):

| flag | default | purpose |
|------|---------|---------|
| `reflect` | `False` | enable post-ranking reflection loop (≤1 extra LLM call) |
| `diversity` | `True` | MMR re-ranking for genre diversity in the top-k |
| `lambda_param` | `0.5` | relevance/diversity tradeoff (1.0 = pure relevance) |

Contrastive explanations — pass `recommended_ids` to include a "why this over
the next-best alternative?" comparison:

```python
client.explain(user_id=196, item_id=64, recommended_ids={64, 12, 77})
```

The comparison finds the user's highest-rated item sharing a genre with the
target that was NOT recommended, and appends a deterministic contrast string.

## Evaluate

```bash
# everything: rating CV + LOO ranking + baselines + bootstrap + 200-user agent
python -m recagent.cli eval --protocol all --baselines --bootstrap --sample 200 --agent

# just ranking, with the long-tail debias (top-2% head targets dropped)
python -m recagent.cli eval --protocol ranking --baselines --exclude-head 0.02

# rating CV only
python -m recagent.cli eval --protocol rating --baselines

# ml-20m: same protocols, plus a deterministic rating subsample for the CV
# (full 5-fold over 20M triples is impractical)
python -m recagent.cli eval --protocol all --data-kind ml-20m --sample-ratings 3000000 \
  --rating-report results/ml20m/eval_rating_ml20m.json
```

All protocols seed their splits (default 42). Reports land in `results/` and
feed `docs/FINDINGS.md`. On ml-20m the LOO protocols should be run with a user
sample (`--sample`) and the rating CV with `--sample-ratings`; the raw
full-scale scripts for the numbers in `docs/FINDINGS.md` §0 are
`scripts/run_ml20m_loo.py`, `scripts/run_ml20m_rating.py`,
`scripts/run_ml20m_t5.py` (classic-model experiments, §0.4),
`scripts/run_t5_probes.py` (svd rating + implicit-alpha probes), and
`scripts/run_ml20m_agent.py` (agent eval on the ml-20m sample — needs a live
vLLM endpoint; `--sample/--seed/--out` control the cohort, default is a
uniform seed-42 200-user sample; the ml-20m-trained `artifacts_ml20m/` and the
ml-100k equivalent `scripts/run_agent200.py` — which also takes
`--exclude-head` (long-tail cohort) and `--no-constraint` — are both runnable
now, see §0.5/§0.6/§0.7/§4; both scripts now support `--exclude-head` and
write per-user agent + ALS rank lists so bootstraps can be recomputed
offline).

**Caching LOO splits** — pass `cache_dir` to `build_test_items` to persist
computed splits to disk. On ml-20m this avoids the ~9 min re-derivation on
every eval build:

```python
from recagent.evaluate import build_test_items
test_items = build_test_items(state, data_dir, data_kind="ml-20m", cache_dir=".cache/splits")
```

Cache files are keyed by `(data_kind, seed, min_interactions, exclude_head)` and
stored as `loo_split_{hash}.json` in the specified directory.

Bootstrap recomputation without any LLM calls:

```bash
# head/tail decomposition of the raw v2 cohort (uses saved per-user ranks)
python scripts/run_agent_headtail.py --artifacts-dir artifacts_als2 \
  --in results/eval_agent200_v2.json --exclude-head 0.2 \
  --out results/eval_agent_headtail.json
```

## Serve

```bash
python -m recagent.cli serve --artifacts artifacts --port 8000
# interactive API docs at http://localhost:8000/docs
```

## SDK

```python
from recagent_sdk import RecommendClient

with RecommendClient("http://127.0.0.1:8000") as client:
    recs = client.recommend(user_id=196, k=5, filters={"genre": "Sci-Fi"})
    why  = client.explain(user_id=196, item_id=64)
```

### Client retry and circuit breaker

`RecClient` retries transient vLLM failures (HTTP 502/503/504, connection
errors) with exponential backoff and jitter. Configure via constructor:

| param | default | purpose |
|-------|---------|---------|
| `max_retries` | 3 | max retry attempts per request |
| `retry_base_delay` | 1.0 | base delay in seconds (doubles each retry, ± jitter) |
| `circuit_threshold` | 5 | consecutive failures before circuit opens |
| `circuit_timeout` | 30 | seconds the circuit stays open before half-open probe |

When the circuit is open, requests fail immediately with a clear error instead
of waiting for timeouts.

## Files that matter

| path | purpose |
|---|---|
| `recagent/` | package: engines, agent, explain, client, api, cli, evaluate, data, config, model, utils, state |
| `recagent/memory.py` | UserMemory: persistent preference buckets (JSON-backed) |
| `recagent/session.py` | SessionMemory: in-conversation context tracking |
| `recagent_sdk/` | typed REST client (sync + async) |
| `results/*.json` | machine-readable eval output (single source of truth) |
| `docs/` | FINDINGS (numbers), DESIGN (why), TESTING (coverage), this runbook |
| `assets/recagent-logo.svg` | mascot logo |
| `data/`, `artifacts*/`, `.env` | gitignored; fetched/trained, never committed |
| `artifacts/memory.json` | user preference storage (auto-created) |

## Production flags

| env var / param | default | what it does |
|---|---|---|
| `--reflect` | `True` | enables reflection + refinement loop |
| `--evidence-budget-tokens` | `4000` | caps evidence text at `budget*4` chars |
| `--lambda-param` | `0.5` | MMR diversity/relevance tradeoff (1.0=pure relevance) |
| `--max-retries` | `3` | exponential backoff retries on transient errors |
| `--retry-base-delay` | `1.0` | base delay in seconds before first retry |
| `request_id` | `None` | correlation ID threaded through to ReasoningTrace |
| `recommended_ids` | `None` | enables contrastive explanations |
| `feedback_path` | `None` | path to feedback JSONL; auto-ingested into memory on init |

## Memory and session

User preferences are stored in `artifacts/memory.json` (auto-created). The
agent sees preference history and current session context in the evidence block
on every call.

### REST endpoints

```
POST /save_preference       {"user_id": 196, "category": "loved", "item_ids": [64, 12]}
POST /get_preferences       {"user_id": 196, "category": "loved"}
POST /get_preference_summary {"user_id": 196}
POST /ingest_feedback       {"feedback_path": "artifacts/feedback.jsonl"}
POST /recommend             {"user_id": 196, "k": 5, "filters": {"mood": "light"}}
```

### SDK

```python
from recagent.client import RecClient

client = RecClient()
client.deps.save_preference(196, "loved", [64, 12], note="favourites")
resp = client.recommend(196, k=5, filters={"mood": "light"})
print(resp.items[0].tags)  # ["comfort", "mood-light"]

# session tracks what was recommended
print(client.session.session_summary())
```

## House rules

- Every merged change keeps `pytest tests/ -q` and `ruff check .` green.
- Evaluation numbers are transcribed from `results/*.json`, never retyped.
- When an experiment disappoints, the disappointment gets written down
  (see the agent-vs-ALS result in `docs/FINDINGS.md` §4).
