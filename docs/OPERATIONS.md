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
python -m recagent.cli train --cf als --data-kind ml-20m   # big dataset (default ml-100k)
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
vLLM endpoint and ml-20m-trained `artifacts/`, see `scripts/run_agent200.py`
for the ml-100k equivalent).

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

## Files that matter

| path | purpose |
|---|---|
| `recagent/` | package: engines, agent, explain, client, api, cli, evaluate, data, config, model |
| `recagent_sdk/` | typed REST client (sync + async) |
| `results/*.json` | machine-readable eval output (single source of truth) |
| `docs/` | FINDINGS (numbers), DESIGN (why), TESTING (coverage), this runbook |
| `assets/recagent-logo.svg` | mascot logo |
| `data/`, `artifacts*/`, `.env` | gitignored; fetched/trained, never committed |

## House rules

- Every merged change keeps `pytest tests/ -q` and `ruff check .` green.
- Evaluation numbers are transcribed from `results/*.json`, never retyped.
- When an experiment disappoints, the disappointment gets written down
  (see the agent-vs-ALS result in `docs/FINDINGS.md` §4).
