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
    iid_to_idx = state["iid_to_idx"]
    items_meta = state["items_meta"]

    user_id = args.user
    if user_id not in uid_to_idx:
        raise SystemExit(f"unknown user id: {user_id}")
    user_idx = uid_to_idx[user_id]
    for item_idx, score in model.recommend(matrix, user_idx, n=args.n):
        item_id = iid_to_idx[item_idx]
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
        genre_share,
        save_report,
    )
    from recagent.state import load_state
    from recagent.tools import ToolRegistry

    state = load_state(args.artifacts)
    test_items = build_test_items(
        state, args.data, min_interactions=args.min_interactions, seed=args.seed
    )
    if args.sample:
        test_items = {u: test_items[u] for u in sorted(test_items)[: args.sample]}

    cf = cf_baseline(state, test_items)
    print(f"CF baseline (ALS) over {cf['n_users']} users")
    for k in cf["hr"]:
        print(f"  HR@{k:<2} {cf['hr'][k]:.4f}   NDCG@{k:<2} {cf['ndcg'][k]:.4f}")

    report = {"dataset": "ml-100k", "cf_baseline": cf, "agent": None}
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
            print(f"  agent genre precision: {agent_share.get(args.genre.lower(), 0.0):.4f}")
            print(f"  CF    genre precision: {cf_share.get(args.genre.lower(), 0.0):.4f}")
            report["constraint"] = {
                "genre": args.genre,
                "agent_precision": agent_share.get(args.genre.lower(), 0.0),
                "cf_precision": cf_share.get(args.genre.lower(), 0.0),
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
    save_report(report, args.report)
    print(f"\nreport -> {args.report}")


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
        "--agent",
        action="store_true",
        help="also run the agentic reranker (requires a vLLM endpoint)",
    )
    ev.add_argument(
        "--genre",
        help="constraint-eval: hold every agent item to this genre, compare genre precision vs CF",
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
    elif args.command == "chat":
        _cmd_chat(args)
    elif args.command == "eval":
        _cmd_eval(args)
    elif args.command == "serve":
        _cmd_serve(args)


if __name__ == "__main__":
    main()
