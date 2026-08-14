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
        },
        args.artifacts,
    )
    n_users, n_items = matrix.shape
    print(f"trained ALS on {n_users} users x {n_items} items")
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
    seen = matrix.getrow(user_idx).indices
    for item_idx, score in model.recommend(matrix, user_idx, n=args.n):
        item_id = iid_to_idx[item_idx]
        info = items_meta.get(item_id, {})
        mark = "seen" if item_idx in seen else ""
        print(f"{info.get('title', item_id):<48} {score:8.4f} {mark}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recagent",
        description="Collaborative filtering candidates, refined by an agentic LLM.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="fetch data and fit the ALS model")
    train.add_argument("--data", default="data", help="data directory (default: data)")
    train.add_argument("--artifacts", default="artifacts", help="output directory")
    train.add_argument("--factors", type=int, default=64)
    train.add_argument("--iterations", type=int, default=20)
    train.add_argument("--min-interactions", type=int, default=5)
    train.add_argument("--seed", type=int, default=42)

    rec = sub.add_parser("recommend", help="top-n recommendations for a user (CF only)")
    rec.add_argument("user", type=int, help="raw user id")
    rec.add_argument("--n", type=int, default=10)
    rec.add_argument("--artifacts", default="artifacts")

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        _cmd_train(args)
    elif args.command == "recommend":
        _cmd_recommend(args)


if __name__ == "__main__":
    main()
