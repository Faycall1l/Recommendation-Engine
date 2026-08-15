"""T5 probes persisted to results/ (house rule: transcribe from JSON, never retype).

1. ml-100k rating CV: svd (BiasedMF, tuned) vs mf (tuned) -> eval_t5_svd_rating_ml100k.json
2. ml-100k implicit-alpha ranking probe (ratings scale the ALS confidence weight)
   -> eval_t5_alpha_ml100k.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recagent.data import encode, fetch_movielens, leave_one_out, load_ratings
from recagent.evaluate import cv_rating_eval_from_arrays
from recagent.model import Recommender

OUT = Path("results/ml20m")


def rating_probe() -> None:
    users, items, ratings = load_ratings(fetch_movielens("data"))
    report = cv_rating_eval_from_arrays(
        users,
        items,
        ratings,
        kinds=("svd", "mf"),
        engine_kwargs={
            "svd": {"factors": 8, "iterations": 20, "regularization": 1.0, "bias_shrinkage": 25.0},
            "mf": {"factors": 6, "iterations": 15, "regularization": 1.0},
        },
        verbose=True,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eval_t5_svd_rating_ml100k.json").write_text(
        json.dumps({"dataset": "ml-100k", "protocol": "5-fold rating CV", "report": report}, indent=2)
    )
    for kind in ("svd", "mf"):
        m = report[kind]
        print(f"rating {kind}  RMSE {m['rmse']:.4f} ± {m['rmse_std']:.4f}  config {m['config']}")


def alpha_probe() -> None:
    users, items, ratings = load_ratings(fetch_movielens("data"))
    (tr_u, tr_i, tr_r), (te_u, te_i) = leave_one_out(users, items, ratings, seed=42)
    matrix, uid_to_idx, _iid_to_idx, _uids, item_ids = encode(tr_u, tr_i, tr_r)
    test_items = {int(u): int(i) for u, i in zip(te_u, te_i) if int(u) in uid_to_idx}
    results: dict[str, dict] = {}
    for scale in (1, 5, 20):
        t0 = time.time()
        scaled = matrix.multiply(scale)
        model = Recommender(factors=64, iterations=20).fit(scaled)
        ranked = {
            u: [int(item_ids[idx]) for idx, _ in model.recommend(scaled, uid_to_idx[u], n=10)]
            for u in test_items
        }
        hits = sum(1 for u in test_items if test_items[u] in ranked[u][:10])
        mrr = sum(
            1.0 / (ranked[u].index(test_items[u]) + 1) if test_items[u] in ranked[u] else 0.0
            for u in test_items
        )
        results[f"alpha{scale}"] = {"hr@10": round(hits / len(test_items), 4), "mrr": round(mrr / len(test_items), 4)}
        print(f"alpha{scale}  HR@10 {results[f'alpha{scale}']['hr@10']}  MRR {results[f'alpha{scale}']['mrr']}  ({time.time()-t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eval_t5_alpha_ml100k.json").write_text(
        json.dumps({"dataset": "ml-100k", "note": "ratings scale the implicit ALS confidence weight",
                    "default_alpha_40": results["alpha1"], "results": results}, indent=2)
    )


if __name__ == "__main__":
    rating_probe()
    alpha_probe()
