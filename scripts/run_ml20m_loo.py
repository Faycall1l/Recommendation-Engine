"""First ml-20m leave-one-out ranking results (T4).

Fits the engines once on the full train matrix (ALS dominates: ~4.5 min),
then scores both the raw holdout and the Cremonesi long-tail holdout
(exclude_head=0.02) on a deterministic 2000-user sample. Writes:
  results/ml20m/eval_ranking_ml20m.json
  results/ml20m/eval_ranking_longtail_ml20m.json

Memory-based user/item CF is intentionally omitted: their similarity products
materialize too many nonzeros at 138k x 27k scale to be tractable even with
the sparse top-k form.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recagent.data import encode, leave_one_out, load_ratings_20m
from recagent.engines import build_engine
from recagent.evaluate import head_item_ids, mean_metrics

DATA_DIR = Path("data/ml-20m")
RESULTS = Path("results/ml20m")
KINDS = ("als", "popular", "random")
K = 10
KS = (1, 3, 5, 10)
USER_SAMPLE = 2000
SEED = 42


def main() -> None:
    t0 = time.time()
    users, items, ratings = load_ratings_20m(DATA_DIR)
    print(f"load {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    (tr_u, tr_i, tr_r), (te_u, te_i) = leave_one_out(users, items, ratings, seed=SEED)
    print(f"loo {time.time() - t0:.1f}s ({len(te_u)} test users)", flush=True)

    t0 = time.time()
    matrix, uid_to_idx, _iid_to_idx, _uids, item_ids = encode(tr_u, tr_i, tr_r)
    print(f"encode {time.time() - t0:.1f}s {matrix.shape}", flush=True)

    t0 = time.time()
    engines = {kind: build_engine(kind, matrix, seed=SEED) for kind in KINDS}
    print(f"fit engines {time.time() - t0:.1f}s", flush=True)

    test_items = {int(u): int(i) for u, i in zip(te_u, te_i) if int(u) in uid_to_idx}
    sample = {u: test_items[u] for u in sorted(test_items)[:USER_SAMPLE]}

    def evaluate(set_name: str, targets: dict[int, int]) -> dict[str, dict]:
        ranked: dict[str, dict[int, list[int]]] = {kind: {} for kind in KINDS}
        for user_id in targets:
            user_idx = uid_to_idx[user_id]
            for kind, engine in engines.items():
                ranked[kind][user_id] = [
                    int(item_ids[idx]) for idx, _ in engine.recommend(matrix, user_idx, n=K)
                ]
        out: dict[str, dict] = {}
        for kind in KINDS:
            metrics = mean_metrics(ranked[kind], targets, KS)
            metrics["kind"] = kind
            out[kind] = metrics
            print(
                f"{set_name:>10} {kind:<8} n={metrics['n_users']} "
                + "  ".join(f"HR@{k}={metrics['hr'][str(k)]:.4f}" for k in KS)
                + f"  NDCG@10={metrics['ndcg']['10']:.4f}  MRR={metrics['mrr']:.4f}",
                flush=True,
            )
        return out

    raw = evaluate("raw", sample)

    head = head_item_ids(items, 0.02)
    long_tail_targets = {u: i for u, i in sample.items() if i not in head}
    print(f"long-tail: dropped {len(sample) - len(long_tail_targets)} head targets", flush=True)
    long_tail = evaluate("longtail", long_tail_targets)

    RESULTS.mkdir(parents=True, exist_ok=True)
    report_raw = {
        "dataset": "ml-20m",
        "protocol": "ranking",
        "seed": SEED,
        "user_sample": USER_SAMPLE,
        "exclude_head": None,
        "engines": raw,
    }
    report_long = {
        "dataset": "ml-20m",
        "protocol": "ranking",
        "seed": SEED,
        "user_sample": USER_SAMPLE,
        "exclude_head": 0.02,
        "engines": long_tail,
    }
    (RESULTS / "eval_ranking_ml20m.json").write_text(json.dumps(report_raw, indent=2))
    (RESULTS / "eval_ranking_longtail_ml20m.json").write_text(json.dumps(report_long, indent=2))
    print(f"\nwrote {RESULTS}/eval_ranking_ml20m.json and eval_ranking_longtail_ml20m.json")


if __name__ == "__main__":
    main()
