"""Command-line interface: recagent train | recommend | chat | eval."""

from __future__ import annotations

import argparse


def _cmd_train(args: argparse.Namespace) -> None:
    from recagent.model import train_from_data
    from recagent.state import save_state

    recommender, matrix, uid_to_idx, iid_to_idx, user_ids, item_ids, items_meta = (
        train_from_data(
            args.data,
            min_interactions=args.min_interactions,
            seed=args.seed,
            factors=args.factors,
            iterations=args.iterations,
            cf=args.cf,
        )
    )
    save_state(
        {
            "model": recommender,
            "matrix": matrix,
            "uid_to_idx": uid_to_idx,
            "iid_to_idx": iid_to_idx,
            "user_ids": user_ids,
            "item_ids": item_ids,
            "items_meta": items_meta,
            "cf_kind": args.cf,
        },
        args.artifacts,
    )
    n_users, n_items = matrix.shape
    print(f"trained {args.cf} CF on {n_users} users x {n_items} items")
    print(f"saved artefacts -> {args.artifacts}")


def _cmd_recommend(args: argparse.Namespace) -> None:
    from recagent.state import load_state

    state = load_state(args.artifacts)
    model = state["model"]
    matrix = state["matrix"]
    uid_to_idx = state["uid_to_idx"]
    items_meta = state["items_meta"]
    item_ids = state["item_ids"]

    user_id = args.user
    if user_id not in uid_to_idx:
        raise SystemExit(f"unknown user id: {user_id}")
    user_idx = uid_to_idx[user_id]
    print(f"engine: {state.get('cf_kind', 'als')}-based CF for user {user_id}")
    for item_idx, score in model.recommend(matrix, user_idx, n=args.n):
        item_id = item_ids[item_idx]
        info = items_meta.get(item_id, {})
        print(f"{info.get('title', item_id):<48} {score:8.4f}")


def _print_trace(result) -> None:
    for message in result.all_messages():
        for part in message.parts:
            kind = part.part_kind
            if kind == "tool-call":
                print(f"  tool-call: {part.tool_name}({part.args})")
            elif kind == "tool-return":
                print(f"  tool-return: {part.tool_name} -> {str(part.content)[:140]}")
            elif kind == "text":
                print(f"  text: {part.content}")


def _print_result(result, deps, verbose: bool) -> None:
    if verbose:
        _print_trace(result)
    output = result.output
    if output is None:
        print("agent produced no ranked list")
        return
    titles = deps.items_meta
    for rank, item in enumerate(output.items, 1):
        title = titles.get(item.item_id, {}).get("title", f"<item {item.item_id}>")
        print(f"{rank:>2}. {title} — {item.reason}")
    from recagent.agent import usage_summary

    print(f"usage: {usage_summary(result)}")


def _cmd_chat(args: argparse.Namespace) -> None:
    from recagent.agent import RecAgent
    from recagent.config import load_llm_config
    from recagent.tools import ToolRegistry

    config = load_llm_config()
    if not config.enabled:
        raise SystemExit(
            "agent disabled — set ATHAR_AGENT__ENABLED=true and point "
            "ATHAR_AGENT__VLLM__BASE_URL / API_KEY / MODEL at your vLLM endpoint"
        )
    deps = ToolRegistry.from_artifacts(args.artifacts)
    agent = RecAgent(config, deps.state)

    def handle(text: str) -> None:
        if args.user is not None:
            text = f"user_id: {args.user}\n\n{text}"
        result = agent.run(text, deps)
        _print_result(result, deps, args.verbose)

    if args.one_shot:
        handle(args.one_shot)
        return
    print(f"recagent chat ({config.model}) — empty line exits")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            return
        handle(line)


def _run_rating_protocol(args: argparse.Namespace) -> str:
    """5-fold CV explicit-rating RMSE/MAE per engine; returns the report path."""
    import json

    from recagent.evaluate import cv_rating_eval

    kinds = ["user", "item", "mf"]
    if args.baselines:
        kinds = ["global-mean", "user-mean", "item-mean", "user", "item", "mf"]
    results = cv_rating_eval(args.data, kinds=kinds, k=args.folds, seed=args.seed)
    print(f"\n5-fold CV explicit-rating prediction (k={args.folds})")
    for kind, m in results.items():
        print(
            f"  {kind:<12} RMSE {m['rmse']:.4f} ± {m['rmse_std']:.4f}   "
            f"MAE {m['mae']:.4f} ± {m['mae_std']:.4f}"
        )
    with open(args.rating_report, "w") as fh:
        json.dump({"protocol": "rating", "folds": args.folds, "engines": results}, fh, indent=2)
    print(f"rating report -> {args.rating_report}")
    return args.rating_report


def _cmd_eval(args: argparse.Namespace) -> None:
    import asyncio

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
        save_report,
    )
    from recagent.state import load_state
    from recagent.tools import ToolRegistry

    if args.protocol in ("rating", "all"):
        _run_rating_protocol(args)
    if args.protocol == "rating":
        return

    state = load_state(args.artifacts)
    test_items = build_test_items(
        state, args.data, min_interactions=args.min_interactions, seed=args.seed
    )
    if args.exclude_head is not None:
        from recagent.data import fetch_movielens, load_ratings
        from recagent.evaluate import head_item_ids

        if not 0.0 <= args.exclude_head < 1.0:
            raise SystemExit(f"--exclude-head must be in [0, 1), got {args.exclude_head}")
        _, items, _ = load_ratings(fetch_movielens(args.data))
        head = head_item_ids(items, args.exclude_head)
        before = len(test_items)
        test_items = {u: i for u, i in test_items.items() if i not in head}
        print(f"excluded {before - len(test_items)} head-item test targets")
    if args.sample:
        test_items = {u: test_items[u] for u in sorted(test_items)[: args.sample]}

    if args.all_cf:
        kinds = ["als", "user", "item"]
    elif args.cf:
        kinds = [args.cf]
    else:
        kinds = [state.get("cf_kind", "als")]

    cf_results: dict[str, dict] = {}
    for kind in kinds:
        cf = cf_baseline(state, test_items, kind=kind)
        cf_results[kind] = cf
        print(f"CF baseline ({kind}) over {cf['n_users']} users")
        for k in cf["hr"]:
            print(f"  HR@{k:<2} {cf['hr'][k]:.4f}   NDCG@{k:<2} {cf['ndcg'][k]:.4f}")

    primary = kinds[0]
    cf = cf_results[primary]
    report = {"dataset": "ml-100k", "cf_baseline": cf_results, "agent": None}
    if args.agent:
        config = load_llm_config()
        if not config.enabled:
            raise SystemExit(
                "agent disabled — set ATHAR_AGENT__ENABLED=true and point "
                "ATHAR_AGENT__VLLM__* at your vLLM endpoint"
            )
        deps = ToolRegistry(state)
        agent = RecAgent(config, state)
        if args.genre:
            users = list(test_items)
            agent_lists, details = asyncio.run(
                constraint_eval(
                    agent, deps, users, constraint=args.genre, k=args.k, concurrency=args.parallel
                )
            )
            cf_top = cf_lists(state, users, n=args.k)
            agent_share = genre_share(state, agent_lists)
            cf_share = genre_share(state, cf_top)
            print(f"\nConstraint compliance ({args.genre}) — CF is genre-blind")
            print(f"  agent genre precision: {genre_precision(agent_share, args.genre):.4f}")
            print(f"  CF    genre precision: {genre_precision(cf_share, args.genre):.4f}")
            report["constraint"] = {
                "genre": args.genre,
                "agent_precision": genre_precision(agent_share, args.genre),
                "cf_precision": genre_precision(cf_share, args.genre),
                "per_user": [
                    {"user_id": u, "cf": ids, "agent": agent_lists.get(u, [])} for u, ids in details
                ],
            }
        else:
            agent_metrics, details = asyncio.run(
                agent_baseline(agent, deps, test_items, k=args.k, concurrency=args.parallel)
            )
            report["agent"] = agent_metrics
            report["per_user"] = details
            print(f"\nAgent (Gemma-4 via pydantic-ai) over {agent_metrics['n_users']} users")
            for k in agent_metrics["hr"]:
                print(
                    f"  HR@{k:<2} {agent_metrics['hr'][k]:.4f}   "
                    f"NDCG@{k:<2} {agent_metrics['ndcg'][k]:.4f}"
                )
            for k in cf["hr"]:
                if k in agent_metrics["hr"]:
                    print(f"  delta HR@{k}  {agent_metrics['hr'][k] - cf['hr'][k]:+.4f}")

    if args.bootstrap:
        from recagent.evaluate import hits_from_ranks, mrr_from_ranks, paired_bootstrap

        present = [k for k in ("als", "user", "item") if k in cf_results]
        report["bootstrap"] = {}
        for i, a in enumerate(present):
            for b in present[i + 1 :]:
                ha = hits_from_ranks(cf_results[a]["per_user_rank"], 5)
                hb = hits_from_ranks(cf_results[b]["per_user_rank"], 5)
                ma = mrr_from_ranks(cf_results[a]["per_user_rank"])
                mb = mrr_from_ranks(cf_results[b]["per_user_rank"])
                key = f"{a}_vs_{b}"
                report["bootstrap"][key] = {
                    "hit_at_5": paired_bootstrap(ha, hb),
                    "mrr": paired_bootstrap(ma, mb),
                }
                hit = report["bootstrap"][key]["hit_at_5"]
                mrr = report["bootstrap"][key]["mrr"]
                print(f"\nPaired bootstrap {a} vs {b} (n={hit['n']})")
                print(
                    f"  hit@5 {hit['mean_diff']:+.4f}  "
                    f"95% CI [{hit['ci_lo']:.4f}, {hit['ci_hi']:.4f}]  p={hit['p_value']:.4f}"
                )
                print(
                    f"  MRR   {mrr['mean_diff']:+.4f}  "
                    f"95% CI [{mrr['ci_lo']:.4f}, {mrr['ci_hi']:.4f}]  p={mrr['p_value']:.4f}"
                )

    save_report(report, args.report)
    print(f"\nreport -> {args.report}")


def _cmd_explain(args: argparse.Namespace) -> None:
    from recagent.config import load_llm_config
    from recagent.explain import RecExplainer, explain_recommendation
    from recagent.tools import ToolRegistry

    deps = ToolRegistry.from_artifacts(args.artifacts)
    explanation = explain_recommendation(deps, args.user, args.item)

    config = load_llm_config()
    if config.enabled:
        text, usage = RecExplainer(config).explain(explanation)
        print(text)
        print(f"usage: {usage}")
    else:
        print(explanation.snippet)
        print("(LLM disabled — deterministic snippet)")

    print(
        f"\nbasis: {explanation.basis}  "
        f"score: {explanation.score}  "
        f"matched genres: {explanation.matched_genres}"
    )
    if explanation.user_likes:
        likes = ", ".join(
            f"{e.title} ({e.rating})" for e in explanation.user_likes if e.rating
        )
        print(f"evidence: liked {likes}")


def _cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from recagent.api import create_app

    app = create_app(args.artifacts)
    uvicorn.run(app, host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recagent",
        description="Collaborative filtering candidates, refined by an agentic LLM.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="fetch data and fit a CF engine")
    train.add_argument("--data", default="data", help="data directory (default: data)")
    train.add_argument("--artifacts", default="artifacts", help="output directory")
    train.add_argument("--factors", type=int, default=64)
    train.add_argument("--iterations", type=int, default=20)
    train.add_argument("--min-interactions", type=int, default=5)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--cf",
        default="user",
        choices=("als", "user", "item"),
        help="collaborative filtering engine (default: user)",
    )

    rec = sub.add_parser("recommend", help="top-n recommendations for a user (CF only)")
    rec.add_argument("user", type=int, help="raw user id")
    rec.add_argument("--n", type=int, default=10)
    rec.add_argument("--artifacts", default="artifacts")

    exp = sub.add_parser(
        "explain", help="why an item was recommended to a user (evidence + prose)"
    )
    exp.add_argument("user", type=int, help="raw user id")
    exp.add_argument("item", type=int, help="raw item id")
    exp.add_argument("--artifacts", default="artifacts")

    chat = sub.add_parser(
        "chat", help="interactive agentic recommender (requires a vLLM endpoint)"
    )
    chat.add_argument("--user", type=int, help="bind the conversation to a user id")
    chat.add_argument("--artifacts", default="artifacts")
    chat.add_argument("--verbose", action="store_true", help="print the tool trace")
    chat.add_argument("--one-shot", help="run a single request and exit")

    ev = sub.add_parser(
        "eval", help="offline eval: CF baseline vs agentic reranker on a holdout"
    )
    ev.add_argument("--data", default="data")
    ev.add_argument("--artifacts", default="artifacts")
    ev.add_argument("--report", default="artifacts/eval_report.json")
    ev.add_argument("--min-interactions", type=int, default=5)
    ev.add_argument("--seed", type=int, default=42)
    ev.add_argument("--k", type=int, default=5, help="items per agent request")
    ev.add_argument("--sample", type=int, help="evaluate the first N users")
    ev.add_argument("--parallel", type=int, default=8, help="concurrent agent requests")
    ev.add_argument(
        "--protocol",
        choices=("rating", "ranking", "all"),
        default="ranking",
        help="rating = 5-fold RMSE/MAE; ranking = LOO holdout; all = both (default: ranking)",
    )
    ev.add_argument("--folds", type=int, default=5, help="folds for the rating protocol")
    ev.add_argument(
        "--rating-report",
        default="artifacts/eval_rating.json",
        help="where to write the rating protocol results",
    )
    ev.add_argument(
        "--baselines",
        action="store_true",
        help="also score trivial baselines (mean/popularity/random) in the protocol",
    )
    ev.add_argument(
        "--bootstrap",
        action="store_true",
        help="paired bootstrap significance between engines on hit@5 and MRR",
    )
    ev.add_argument(
        "--agent",
        action="store_true",
        help="also run the agentic reranker (requires a vLLM endpoint)",
    )
    ev.add_argument(
        "--cf",
        choices=("als", "user", "item"),
        help="CF engine for the baseline (default: the engine the artefacts were trained with)",
    )
    ev.add_argument(
        "--all-cf",
        action="store_true",
        help="run the baseline with every engine (als, user, item)",
    )
    ev.add_argument(
        "--genre",
        help="constraint-eval: hold every agent item to this genre, compare genre precision vs CF",
    )
    ev.add_argument(
        "--exclude-head",
        type=float,
        help="Cremonesi debias: drop test targets in the top X fraction of most-popular items",
    )

    serve = sub.add_parser(
        "serve", help="run the REST gateway (FastAPI) for the pipeline"
    )
    serve.add_argument("--artifacts", default="artifacts")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        _cmd_train(args)
    elif args.command == "recommend":
        _cmd_recommend(args)
    elif args.command == "explain":
        _cmd_explain(args)
    elif args.command == "chat":
        _cmd_chat(args)
    elif args.command == "eval":
        _cmd_eval(args)
    elif args.command == "serve":
        _cmd_serve(args)


if __name__ == "__main__":
    main()
