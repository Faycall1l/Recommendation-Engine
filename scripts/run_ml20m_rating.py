"""First ml-20m 5-fold rating-CV results (T4).

Runs cv_rating_eval_from_arrays on a deterministic 3M-rating subsample of
ml-20m (full 5-fold CV over 20M triples is impractical). mf (ExplicitALS) plus
the mean baselines — memory-based user/item are omitted because their
similarity matrices are untractable at this scale. Writes:
  results/ml20m/eval_rating_ml20m.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recagent.data import load_ratings_20m
from recagent.evaluate import cv_rating_eval_from_arrays

RESULTS = Path("results/ml20m")
KINDS = ("mf", "global-mean", "user-mean", "item-mean")
SAMPLE_RATINGS = 3_000_000
FOLDS = 5
SEED = 42


def main() -> None:
    t0 = time.time()
    users, items, ratings = load_ratings_20m(Path("data/ml-20m"))
    print(f"load {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    results = cv_rating_eval_from_arrays(
        users,
        items,
        ratings,
        kinds=KINDS,
        k=FOLDS,
        seed=SEED,
        sample_ratings=SAMPLE_RATINGS,
        verbose=True,
    )
    print(f"cv {time.time() - t0:.1f}s", flush=True)
    for kind, m in results.items():
        print(
            f"  {kind:<12} RMSE {m['rmse']:.4f} ± {m['rmse_std']:.4f}   "
            f"MAE {m['mae']:.4f} ± {m['mae_std']:.4f}",
            flush=True,
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": "ml-20m",
        "protocol": "rating",
        "folds": FOLDS,
        "seed": SEED,
        "sample_ratings": SAMPLE_RATINGS,
        "engines": results,
    }
    (RESULTS / "eval_rating_ml20m.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {RESULTS}/eval_rating_ml20m.json")


if __name__ == "__main__":
    main()
