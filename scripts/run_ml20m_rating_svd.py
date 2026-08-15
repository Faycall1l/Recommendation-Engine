"""T5 rating extension: ml-20m rating CV including BiasedMF (svd).

Same deterministic 3M-rating subsample / 5-fold CV as run_ml20m_rating.py
(so the mf + mean rows reproduce), adding the tuned svd config. Writes:
  results/ml20m/eval_rating_ml20m_svd.json
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
KINDS = ("svd", "mf", "global-mean", "user-mean", "item-mean")
SAMPLE_RATINGS = 3_000_000
FOLDS = 5
SEED = 42
ENGINE_KWARGS = {
    "svd": {"factors": 8, "iterations": 20, "regularization": 1.0, "bias_shrinkage": 25.0},
    "mf": {"factors": 6, "iterations": 15, "regularization": 1.0},
}


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
        engine_kwargs=ENGINE_KWARGS,
        verbose=True,
    )
    print(f"cv {time.time() - t0:.1f}s", flush=True)
    for kind, m in results.items():
        print(
            f"  {kind:<12} RMSE {m['rmse']:.4f} ± {m['rmse_std']:.4f}   "
            f"MAE {m['mae']:.4f} ± {m['mae_std']:.4f}   config={m.get('config')}",
            flush=True,
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": "ml-20m",
        "protocol": "rating",
        "folds": FOLDS,
        "seed": SEED,
        "sample_ratings": SAMPLE_RATINGS,
        "engine_kwargs": ENGINE_KWARGS,
        "engines": results,
    }
    (RESULTS / "eval_rating_ml20m_svd.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {RESULTS}/eval_rating_ml20m_svd.json")


if __name__ == "__main__":
    main()
