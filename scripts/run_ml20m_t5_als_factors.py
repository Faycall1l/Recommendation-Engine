"""T5 confirm: ALS factor-count tuning lifts ml-20m ranking.

ml-100k probing showed factors=24-32 beats the 64 default for LOO ranking.
Confirms on ml-20m (2000-user sample, raw + long-tail) with the tuned configs
and a RankBlend on the best, saving:
  results/ml20m/eval_t5_als_factors_ml20m.json
  results/ml20m/eval_t5_als_factors_longtail_ml20m.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recagent.baselines import MostPopular
from recagent.blend import RankBlend
from recagent.data import encode, leave_one_out, load_ratings_20m
from recagent.engines import build_engine
from recagent.evaluate import head_item_ids, mean_metrics

DATA_DIR = Path("data/ml-20m")
RESULTS = Path("results/ml20m")
K = 10
KS = (1, 3, 5, 10)
USER_SAMPLE = 2000
SEED = 42
FACTORS = (24, 32)


def main() -> None:
    t0 = time.time()
    users, items, ratings = load_ratings_20m(DATA_DIR)
    (tr_u, tr_i, tr_r), (te_u, te_i) = leave_one_out(users, items, ratings, seed=SEED)
    matrix, uid_to_idx, _iid_to_idx, _uids, item_ids = encode(tr_u, tr_i, tr_r)
    print(f"encode {matrix.shape} {time.time() - t0:.1f}s", flush=True)

    engines: dict[str, object] = {}
    for factors in FACTORS:
        name = f"als_f{factors}"
        t0 = time.time()
        engines[name] = build_engine("als", matrix, seed=SEED, factors=factors)
        print(f"{name} fit {time.time() - t0:.1f}s", flush=True)
    best = f"als_f{FACTORS[0]}"
    blend = RankBlend(base_kind="als", base_weight=0.7, top_k=200, k=60, seed=SEED)
    blend.base = engines[best]
    blend.popularity = MostPopular().fit(matrix)
    engines[f"blend_{best}_0.7"] = blend

    test_items = {int(u): int(i) for u, i in zip(te_u, te_i) if int(u) in uid_to_idx}
    sample = {u: test_items[u] for u in sorted(test_items)[:USER_SAMPLE]}

    def evaluate(set_name: str, targets: dict[int, int]) -> dict[str, dict]:
        ranked: dict[str, dict[int, list[int]]] = {name: {} for name in engines}
        for user_id in targets:
            user_idx = uid_to_idx[user_id]
            for name, engine in engines.items():
                ranked[name][user_id] = [
                    int(item_ids[idx]) for idx, _ in engine.recommend(matrix, user_idx, n=K)
                ]
        out: dict[str, dict] = {}
        for name in engines:
            metrics = mean_metrics(ranked[name], targets, KS)
            metrics["kind"] = name
            out[name] = metrics
            print(
                f"{set_name:>4} {name:<18} n={metrics['n_users']} "
                + "  ".join(f"HR@{k}={metrics['hr'][str(k)]:.4f}" for k in KS)
                + f"  NDCG@10={metrics['ndcg']['10']:.4f}  MRR={metrics['mrr']:.4f}",
                flush=True,
            )
        return out

    raw = evaluate("raw", sample)
    head = head_item_ids(items, 0.02)
    long_tail_targets = {u: i for u, i in sample.items() if i not in head}
    long_tail = evaluate("tail", long_tail_targets)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "eval_t5_als_factors_ml20m.json").write_text(
        json.dumps({"dataset": "ml-20m", "seed": SEED, "user_sample": USER_SAMPLE,
                    "engines": raw}, indent=2)
    )
    (RESULTS / "eval_t5_als_factors_longtail_ml20m.json").write_text(
        json.dumps({"dataset": "ml-20m", "seed": SEED, "user_sample": USER_SAMPLE,
                    "exclude_head": 0.02, "engines": long_tail}, indent=2)
    )
    print(f"\nwrote {RESULTS}/eval_t5_als_factors_ml20m.json")


if __name__ == "__main__":
    main()
