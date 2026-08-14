"""Reproducible long-tail LOO ranking eval on ml-100k.

Runs the full 943-user leave-one-out ranking battery with the Cremonesi–
Koren–Turrin (2010) debias: test targets that fall in the top ``EXCLUDE_HEAD``
fraction of most-popular items are dropped. Writes
``results/eval_ranking_longtail.json``.

    python -m scripts.run_longtail_loo
"""

import json
from pathlib import Path

from recagent.evaluate import loo_ranking_eval

EXCLUDE_HEAD = 0.02  # top 2% of most-popular items (Cremonesi et al. 2010)
KINDS = ["als", "user", "item", "mf", "popular", "random"]
ENGINE_KWARGS = {"mf": {"factors": 6, "iterations": 15, "regularization": 1.0}}
OUT = Path("results") / "eval_ranking_longtail.json"


def main() -> None:
    results = loo_ranking_eval(
        "data",
        kinds=KINDS,
        exclude_head=EXCLUDE_HEAD,
        engine_kwargs=ENGINE_KWARGS,
    )
    print(f"long-tail LOO ranking (exclude_head={EXCLUDE_HEAD})")
    for kind, metrics in results.items():
        print(
            f"  {kind:<8} n={metrics['n_users']:<4} "
            f"HR@5 {metrics['hr']['5']:.4f}  HR@10 {metrics['hr']['10']:.4f}  "
            f"NDCG@10 {metrics['ndcg']['10']:.4f}  MRR {metrics['mrr']:.4f}"
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nreport -> {OUT}")


if __name__ == "__main__":
    main()
