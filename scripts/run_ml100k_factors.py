"""T5 probe persisted: ml-100k ALS factor-count sweep for LOO ranking.

The ml-20m sweep (eval_t5_als_factors_ml20m.json) shows f24/f32 losing to f64;
this file records the ml-100k side where fewer factors win, so FINDINGS §0.4's
"tuning does not transfer" claim has both datasets in results/. Writes:
  results/ml20m/eval_t5_als_factors_ml100k.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recagent.data import fetch_movielens, load_ratings
from recagent.evaluate import loo_ranking_eval_from_arrays

OUT = Path("results/ml20m")


def main() -> None:
    users, items, ratings = load_ratings(fetch_movielens("data"))
    report: dict[str, dict] = {}
    for factors in (24, 32, 64):
        t0 = time.time()
        results = loo_ranking_eval_from_arrays(
            users, items, ratings, kinds=("als",), factors=factors, iterations=20
        )
        m = results["als"]
        report[f"f{factors}"] = {
            "hr": m["hr"],
            "ndcg": m["ndcg"],
            "mrr": m["mrr"],
            "n_users": m["n_users"],
        }
        print(
            f"f{factors}  HR@10 {m['hr']['10']}  HR@5 {m['hr']['5']}  "
            f"MRR {m['mrr']}  ({time.time() - t0:.1f}s)",
            flush=True,
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eval_t5_als_factors_ml100k.json").write_text(
        json.dumps(
            {"dataset": "ml-100k", "protocol": "LOO ranking (full 943 users)",
             "seed": 42, "factors": report},
            indent=2,
        )
    )
    print(f"wrote {OUT}/eval_t5_als_factors_ml100k.json")


if __name__ == "__main__":
    main()
