# Testing

**136 tests** across 16 files, green in ~3s with no network. Run with:

```bash
.venv/bin/python -m pytest tests/ -q        # full suite
.venv/bin/python -m pytest tests/test_cf.py # one file
.venv/bin/ruff check .                       # linter (must be clean)
```

## Test inventory

| file | tests | what it guards |
|---|---|---|
| `tests/test_cf.py` | 20 | memory-based CF math: Pearson hand cases, adjusted cosine, min-sim flooring, fallback-to-mean, score_all ≡ predict, exclude-rated/cap-n, save/load round-trips |
| `tests/test_evaluate.py` | 24 | metrics (RMSE/MAE, hits, HR, NDCG, MRR), CV & LOO protocols, engine/baseline wiring, paired bootstrap (finds separation, null stays insignificant, deterministic), genre precision, `head_item_ids` + `exclude_head` validation |
| `tests/test_client.py` | 13 | `RecClient` facade: CF degradation without `.env`, cold-start popularity prior, fake-agent metadata mapping, filters→constraints, feedback appends JSONL, explain API, both engines |
| `tests/test_agent.py` | 9 | plan parsing, evidence builders (warm/cold/rare-genre), genre alias detection, hallucination/constraint/dup guardrails, no-network construction |
| `tests/test_api.py` | 7 | REST gateway: health, recommend (+bad k rejected), chat, explain, feedback, catalog |
| `tests/test_data.py` | 8 | encoding, pinned index space, LOO disjointness, split partitioning/coverage/determinism, cold-user skipping |
| `tests/test_cli.py` | 8 | train flags, recommend prints engine kind, explain prints evidence, eval flags + `--exclude-head`, rating protocol writes a report |
| `tests/test_sdk.py` | 6 | SDK typed models, error raising, async client, explain + feedback + catalog |
| `tests/test_tools.py` | 7 | ToolRegistry: recommend excludes seen + returns meta, item info/profile, genre search/filter, similar excludes seed, trending prior |
| `tests/test_explain.py` | 7 | rating-weighted genres, each basis (genre-affinity, similar-evidence, taste-overlap, cold-start), grounded restatement, empty-LLM fallback to snippet |
| `tests/test_baselines.py` | 5 | global/user/item mean, most-popular rankings, deterministic random |
| `tests/test_mf.py` | 5 | ALS recovers low-rank structure, predict ≡ dot product, recommend sort/exclude, determinism, round-trip |
| `tests/test_engines.py` | 5 | engine factory, kind groupings, rating-vs-ranking kinds |
| `tests/test_model.py` | 3 | model fit/recommend, similar_items self-first, round-trip |

## Patterns the suite relies on

- **Fake agents, not live LLMs.** `.env` is gitignored, so CI never hits a
  model. Tests that need an agent inject a stub whose output is controlled
  (`tests/test_client.py`, `tests/test_explain.py`). Any test that might reach
  a real LLM path monkeypatches `recagent.config.load_llm_config`
  (`test_cli.py::test_explain_prints_evidence`).
- **Hand-computed expectations.** `test_userbased_similarity_pearson_hand_case`
  and friends encode the algebra by hand, so a refactor that changes math
  fails loudly.
- **Tiny synthetic matrices.** Every eval test uses arrays so small the answer
  is obvious; the one degenerate case (adjusted-cosine needs co-rated variance)
  is documented at the test.
- **Round-trip everything.** Every engine and model has save/load tests;
  `State` is engine-agnostic.
- **Determinism.** Same seed ⇒ identical output asserted directly.

## Live verification (optional, needs `.env` + trained artefacts)

Used during development to confirm behaviour against the real vLLM gateway:

```bash
.venv/bin/python -m recagent.cli serve --artifacts artifacts --port 8087 &
curl -s localhost:8087/health
.venv/bin/python - <<'PY'
from recagent_sdk import RecommendClient
c = RecommendClient("http://127.0.0.1:8087")
print(c.recommend(user_id=196, k=5, filters={"genre": "Sci-Fi"}))
print(c.explain(user_id=196, item_id=64))
PY
kill %1
```

Never commit `.env` or `artifacts/`; both are gitignored and regenerable.
