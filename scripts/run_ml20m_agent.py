"""T6 agent eval on the ml-20m LOO sample (raw + long-tail + constraint).

Mirrors run_agent200.py but against ml-20m data and, optionally, a separate
artifacts dir trained on ml-20m. Requires:
  - a live vLLM LLM endpoint (see recagent/config.py / load_llm_config)
  - an artifacts dir whose uid space covers the ml-20m sample users
Writes results/ml20m/eval_agent200_ml20m.json.

Run:  python scripts/run_ml20m_agent.py [artifacts_dir]
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recagent.agent import RecAgent
from recagent.config import load_llm_config
from recagent.evaluate import (
    agent_baseline,
    aligned_rank_arrays,
    build_test_items,
    cf_baseline,
    cf_lists,
    constraint_eval,
    genre_precision,
    genre_share,
    paired_bootstrap,
    save_report,
)
from recagent.state import load_state
from recagent.tools import ToolRegistry

SAMPLE = 200
SEED = 42
OUT = Path("results/ml20m/eval_agent200_ml20m.json")


def main(
    artifacts_dir: str,
    sample: int = SAMPLE,
    seed: int = SEED,
    exclude_head: float | None = None,
    out: str | None = None,
    no_constraint: bool = False,
) -> None:
    state = load_state(artifacts_dir)
    test_items = build_test_items(
        state,
        "data",
        min_interactions=5,
        seed=seed,
        data_kind="ml-20m",
        exclude_head=exclude_head,
    )
    rng = random.Random(seed)
    users = rng.sample(sorted(test_items), k=min(sample, len(test_items)))
    test_items = {u: test_items[u] for u in users}

    als = cf_baseline(state, test_items, kind="als", factors=24)
    print(f"als over {als['n_users']} users: HR@5 {als['hr']['5']:.4f} MRR {als['mrr']:.4f}")

    config = load_llm_config()
    agent = RecAgent(config, state)
    deps = ToolRegistry(state)

    agent_metrics, details = asyncio.run(
        agent_baseline(agent, deps, test_items, k=5, concurrency=8)
    )
    print(
        f"agent over {agent_metrics['n_users']} users: "
        f"HR@5 {agent_metrics['hr']['5']:.4f} MRR {agent_metrics['mrr']:.4f}"
    )
    agent_ranks = {u: _rank(ids, test_items[u]) for u, _, ids in details if u in test_items}

    users = list(test_items)
    constraint_block = None
    if not no_constraint:
        agent_lists, _cdetails = asyncio.run(
            constraint_eval(agent, deps, users, constraint="sci-fi", k=5, concurrency=8)
        )
        cf_top = cf_lists(state, users, n=5)
        agent_share = genre_share(state, agent_lists)
        cf_share = genre_share(state, cf_top)
        agent_prec = genre_precision(agent_share, "sci-fi")
        cf_prec = genre_precision(cf_share, "sci-fi")
        print(f"sci-fi precision  agent {agent_prec:.4f}  cf {cf_prec:.4f}")
        constraint_block = {
            "genre": "sci-fi",
            "agent_precision": genre_precision(agent_share, "sci-fi"),
            "cf_precision": genre_precision(cf_share, "sci-fi"),
            "per_user": [
                {"user_id": u, "cf": cf_top.get(u, []), "agent": agent_lists.get(u, [])}
                for u in users
            ],
        }

    boot = None
    if als["per_user_rank"] and agent_ranks:
        _ha, _hg, _ma, _mg, _uids = aligned_rank_arrays(
            als["per_user_rank"], agent_ranks, 5
        )
        boot = {
            "agent_vs_als_hit5": paired_bootstrap(_hg, _ha),
            "agent_vs_als_mrr": paired_bootstrap(_mg, _ma),
        }
        b = boot["agent_vs_als_hit5"]
        print(
            f"agent vs als hit@5 {b['mean_diff']:+.4f} "
            f"[{b['ci_lo']:.3f},{b['ci_hi']:.3f}] p={b['p_value']:.4f}"
        )

    report = {
        "dataset": "ml-20m",
        "sample": len(users),
        "seed": seed,
        "cohort": (
            "uniform random sample of LOO users (was first-200-by-id)"
            if exclude_head is None
            else f"long-tail uniform seed-{seed} sample (exclude_head={exclude_head})"
        ),
        "exclude_head": exclude_head,
        "k": 5,
        "als": {k2: v for k2, v in als.items() if k2 != "per_user_rank"},
        "als_per_user_rank": {str(u): r for u, r in als["per_user_rank"].items()},
        "agent": agent_metrics,
        "agent_per_user_rank": {str(u): r for u, r in agent_ranks.items()},
        "per_user": details,
        "constraint": constraint_block,
        "bootstrap": boot,
        "context_engineering": "v2 (rating scale, per-item popularity/avg, social proof)",
    }
    out_path = Path(out) if out else OUT
    save_report(report, out_path)
    print(f"wrote {out_path}")


def _rank(ids: list[int], target: int) -> int:
    try:
        return ids.index(target) + 1
    except ValueError:
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts_dir", nargs="?", default="artifacts")
    parser.add_argument("--sample", type=int, default=SAMPLE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--exclude-head", type=float, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--no-constraint", action="store_true")
    args = parser.parse_args()
    main(
        args.artifacts_dir,
        sample=args.sample,
        seed=args.seed,
        exclude_head=args.exclude_head,
        out=args.out,
        no_constraint=args.no_constraint,
    )
