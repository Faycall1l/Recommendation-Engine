"""Head-vs-tail decomposition of the ml-100k raw agent cohort (no LLM calls).

Splits the existing `results/eval_agent200_v2.json` 200-user cohort by whether
the held-out LOO target is among the top ``--exclude-head`` fraction of
most-popular items, then recomputes agent-vs-ALS HR@5/MRR and paired
bootstrap within each subset. Tests the §0.7 hypothesis that the agent's
raw-ranking deficit concentrates on the long tail.

Writes results/eval_agent_headtail.json.

Run:  python scripts/run_agent_headtail.py [--exclude-head 0.2]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from recagent.data import loaders
from recagent.evaluate import (
    head_item_ids,
    hits_from_ranks,
    mrr_from_ranks,
    paired_bootstrap,
    save_report,
)

SRC = Path("results/eval_agent200_v2.json")
OUT = Path("results/eval_agent_headtail.json")


def main(exclude_head: float) -> None:
    with open(SRC) as fh:
        d = json.load(fh)
    test_items = {int(u): int(t) for u, t, _ in d["per_user"]}
    agent_ranks = {int(u): v for u, v in d["agent_per_user_rank"].items()}
    als_ranks = {int(u): v for u, v in d["als_per_user_rank"].items()}

    fetch, load_ratings_fn, _ = loaders("ml-100k")
    dataset_dir = fetch("data")
    _users, items, _ratings = load_ratings_fn(dataset_dir)
    head = head_item_ids(items, exclude_head)

    tail_users = [u for u, t in test_items.items() if t not in head]
    head_users = [u for u, t in test_items.items() if t in head]
    print(f"cohort {len(test_items)} users: {len(head_users)} head-target, {len(tail_users)} tail-target")

    rows = {}
    for label, uids in (("head", head_users), ("tail", tail_users), ("all", list(test_items))):
        if not uids:
            continue
        h_a = hits_from_ranks({u: agent_ranks[u] for u in uids}, 5)
        h_b = hits_from_ranks({u: als_ranks[u] for u in uids}, 5)
        m_a = mrr_from_ranks({u: agent_ranks[u] for u in uids})
        m_b = mrr_from_ranks({u: als_ranks[u] for u in uids})
        rows[label] = {
            "n": len(uids),
            "agent_hr5": float(np.mean(h_a)),
            "als_hr5": float(np.mean(h_b)),
            "agent_mrr": float(np.mean(m_a)),
            "als_mrr": float(np.mean(m_b)),
            "agent_vs_als_hit5": paired_bootstrap(h_a, h_b),
            "agent_vs_als_mrr": paired_bootstrap(m_a, m_b),
        }
        b = rows[label]["agent_vs_als_hit5"]
        print(
            f"{label:4s} n={len(uids):3d} agent {rows[label]['agent_hr5']:.4f} "
            f"als {rows[label]['als_hr5']:.4f} "
            f"hit5 {b['mean_diff']:+.4f} [{b['ci_lo']:.3f},{b['ci_hi']:.3f}] p={b['p_value']:.4f}"
        )

    report = {
        "source": str(SRC),
        "exclude_head": exclude_head,
        "n_head_items": len(head),
        "n_users": len(test_items),
        "n_head_targets": len(head_users),
        "n_tail_targets": len(tail_users),
        "subsets": rows,
        "note": "decomposes the raw ml-100k context-v2 cohort; no LLM calls",
    }
    save_report(report, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclude-head", type=float, default=0.2)
    args = parser.parse_args()
    main(args.exclude_head)
