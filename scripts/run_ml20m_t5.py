"""T5: classic recSys improvements on ml-20m.

Fits the strong personalized engines once (ALS + BiasedMF) and evaluates:
  - als, svd, popular (reference)
  - RankBlend{als base, weights 0.3/0.5/0.7} fusing ALS with the popularity prior
on both the raw and debiased long-tail 2000-user LOO holdouts. Writes:
  results/ml20m/eval_t5_ranking_ml20m.json
  results/ml20m/eval_t5_ranking_longtail_ml20m.json

Blend engines reuse the single ALS fit (their recommend() only needs the base's
rank list), so this is one ALS fit + one BiasedMF fit total.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import scipy.sparse as sp

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
BLEND_WEIGHTS = (0.3, 0.5, 0.7)
SVD_KWARGS = {"factors": 32, "iterations": 20, "regularization": 0.1}


def build_blends(matrix: sp.csr_matrix, als_engine) -> dict[str, RankBlend]:
    """RankBlends reusing the fitted ALS engine (no refit per weight)."""
    popularity = MostPopular().fit(matrix)
    blends: dict[str, RankBlend] = {}
    for weight in BLEND_WEIGHTS:
        blend = RankBlend(base_kind="als", base_weight=weight, top_k=200, k=60, seed=SEED)
        blend.base = als_engine
        blend.popularity = popularity
        blends[f"blend{weight:.1f}"] = blend
    return blends


def main() -> None:
    t0 = time.time()
    users, items, ratings = load_ratings_20m(DATA_DIR)
    print(f"load {time.time() - t0:.1f}s", flush=True)
    (tr_u, tr_i, tr_r), (te_u, te_i) = leave_one_out(users, items, ratings, seed=SEED)
    matrix, uid_to_idx, _iid_to_idx, _uids, item_ids = encode(tr_u, tr_i, tr_r)
    print(f"encode {matrix.shape} {time.time() - t0:.1f}s", flush=True)

    print("fitting als...", flush=True)
    als_engine = build_engine("als", matrix, seed=SEED)
    print(f"als fit {time.time() - t0:.1f}s", flush=True)
    print("fitting svd...", flush=True)
    svd_engine = build_engine("svd", matrix, seed=SEED, **SVD_KWARGS)
    print(f"svd fit {time.time() - t0:.1f}s", flush=True)
    popular_engine = build_engine("popular", matrix, seed=SEED)

    engines: dict[str, object] = {
        "als": als_engine,
        "svd": svd_engine,
        "popular": popular_engine,
    }
    engines.update(build_blends(matrix, als_engine))
    print(f"engines ready {time.time() - t0:.1f}s", flush=True)

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
        for name, metrics in ((name, mean_metrics(ranked[name], targets, KS)) for name in engines):
            metrics["kind"] = name
            out[name] = metrics
            print(
                f"{set_name:>8} {name:<10} n={metrics['n_users']} "
                + "  ".join(f"HR@{k}={metrics['hr'][str(k)]:.4f}" for k in KS)
                + f"  NDCG@10={metrics['ndcg']['10']:.4f}  MRR={metrics['mrr']:.4f}",
                flush=True,
            )
        return out

    raw = evaluate("raw", sample)
    head = head_item_ids(items, 0.02)
    long_tail_targets = {u: i for u, i in sample.items() if i not in head}
    print(f"long-tail: dropped {len(sample) - len(long_tail_targets)} head targets", flush=True)
    long_tail = evaluate("tail", long_tail_targets)

    RESULTS.mkdir(parents=True, exist_ok=True)
    base_report = {
        "dataset": "ml-20m",
        "protocol": "ranking",
        "seed": SEED,
        "user_sample": USER_SAMPLE,
        "svd_kwargs": SVD_KWARGS,
        "blend_top_k": 200,
        "blend_k": 60,
        "engines": raw,
    }
    (RESULTS / "eval_t5_ranking_ml20m.json").write_text(json.dumps(base_report, indent=2))
    long_report = {**base_report, "exclude_head": 0.02, "engines": long_tail}
    (RESULTS / "eval_t5_ranking_longtail_ml20m.json").write_text(
        json.dumps(long_report, indent=2)
    )
    print(f"\nwrote {RESULTS}/eval_t5_ranking_ml20m.json and eval_t5_ranking_longtail_ml20m.json")


if __name__ == "__main__":
    main()
