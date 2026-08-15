import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recagent.agent import RecAgent
from recagent.config import load_llm_config
from recagent.evaluate import (
    agent_baseline,
    build_test_items,
    cf_baseline,
    cf_lists,
    constraint_eval,
    genre_precision,
    genre_share,
    hits_from_ranks,
    mrr_from_ranks,
    paired_bootstrap,
    save_report,
)
from recagent.state import load_state
from recagent.tools import ToolRegistry

SAMPLE = 200
OUT = Path("results/eval_agent200_v2.json")


def main(artifacts_dir: str = "artifacts") -> None:
    state = load_state(artifacts_dir)
    test_items = build_test_items(state, "data", min_interactions=5, seed=42)
    test_items = {u: test_items[u] for u in sorted(test_items)[:SAMPLE]}

    als = cf_baseline(state, test_items, kind="als")
    print(f"als over {als['n_users']} users: HR@5 {als['hr']['5']:.4f} MRR {als['mrr']:.4f}")

    config = load_llm_config()
    agent = RecAgent(config, state)
    deps = ToolRegistry(state)

    agent_metrics, details = asyncio.run(
        agent_baseline(agent, deps, test_items, k=5, concurrency=8)
    )
    print(f"agent over {agent_metrics['n_users']} users: HR@5 {agent_metrics['hr']['5']:.4f} MRR {agent_metrics['mrr']:.4f}")
    agent_ranks = {u: _rank(ids, test_items[u]) for u, _, ids in details if u in test_items}

    users = list(test_items)
    agent_lists, _cdetails = asyncio.run(
        constraint_eval(agent, deps, users, constraint="sci-fi", k=5, concurrency=8)
    )
    cf_top = cf_lists(state, users, n=5)
    agent_share = genre_share(state, agent_lists)
    cf_share = genre_share(state, cf_top)
    agent_prec = genre_precision(agent_share, "sci-fi")
    cf_prec = genre_precision(cf_share, "sci-fi")
    print(f"sci-fi precision  agent {agent_prec:.4f}  cf {cf_prec:.4f}")

    boot = None
    if als["per_user_rank"] and agent_ranks:
        ha = hits_from_ranks(als["per_user_rank"], 5)
        hg = hits_from_ranks(agent_ranks, 5)
        ma = mrr_from_ranks(als["per_user_rank"])
        mg = mrr_from_ranks(agent_ranks)
        boot = {
            "agent_vs_als_hit5": paired_bootstrap(hg, ha),
            "agent_vs_als_mrr": paired_bootstrap(mg, ma),
        }
        b = boot["agent_vs_als_hit5"]
        print(f"agent vs als hit@5 {b['mean_diff']:+.4f} [{b['ci_lo']:.3f},{b['ci_hi']:.3f}] p={b['p_value']:.4f}")

    report = {
        "dataset": "ml-100k",
        "sample": SAMPLE,
        "k": 5,
        "als": {k2: v for k2, v in als.items() if k2 != "per_user_rank"},
        "als_per_user_rank": {str(u): r for u, r in als["per_user_rank"].items()},
        "agent": agent_metrics,
        "agent_per_user_rank": {str(u): r for u, r in agent_ranks.items()},
        "per_user": details,
        "constraint": {
            "genre": "sci-fi",
            "agent_precision": genre_precision(agent_share, "sci-fi"),
            "cf_precision": genre_precision(cf_share, "sci-fi"),
            "per_user": [
                {"user_id": u, "cf": cf_top.get(u, []), "agent": agent_lists.get(u, [])}
                for u in users
            ],
        },
        "bootstrap": boot,
        "context_engineering": "v2 (rating scale, per-item popularity/avg, social proof)",
    }
    save_report(report, OUT)
    print(f"wrote {OUT}")


def _rank(ids, target):
    try:
        return ids.index(target) + 1
    except ValueError:
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts_dir", nargs="?", default="artifacts")
    args = parser.parse_args()
    main(args.artifacts_dir)
