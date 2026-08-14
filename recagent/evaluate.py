"""Offline evaluation: does the agent beat the raw collaborative filter?

Leave-one-out holdout (same split as training) scored with hit-rate and NDCG.
The agent is scored on the exact lists it emits, so a better result is evidence
the reasoning+tools stage adds signal beyond ALS alone.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from recagent.agent import RecAgent
from recagent.data import encode, fetch_movielens, leave_one_out, load_ratings, split_ratings
from recagent.tools import ToolRegistry

KS = (1, 3, 5, 10)


def rating_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """RMSE/MAE over a batch of (actual, predicted) explicit ratings."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) == 0:
        return {"rmse": 0.0, "mae": 0.0, "n": 0}
    errors = actual - predicted
    return {
        "rmse": round(float(np.sqrt(np.mean(errors**2))), 4),
        "mae": round(float(np.mean(np.abs(errors))), 4),
        "n": len(actual),
    }


def cv_rating_eval_from_arrays(
    users: np.ndarray,
    items: np.ndarray,
    ratings: np.ndarray,
    *,
    kinds: Sequence[str] | None = None,
    k: int = 5,
    seed: int = 42,
    factors: int = 64,
    iterations: int = 20,
) -> dict:
    """5-fold CV explicit-rating prediction: RMSE/MAE mean+-std per engine.

    Engines are refit from scratch on each fold's train matrix and scored on
    the fold's held-out triples. ``als`` is rejected (no calibrated predict).
    """
    from recagent.engines import RATING_ENGINES, build_engine

    kinds = list(kinds or RATING_ENGINES)
    for kind in kinds:
        if kind not in RATING_ENGINES:
            raise ValueError(f"rating protocol supports {RATING_ENGINES}, got {kind!r}")
    matrix, uid_to_idx, iid_to_idx, _user_ids, _item_ids = encode(users, items, ratings)
    folds = split_ratings(users, items, ratings, k=k, seed=seed)
    per_fold: dict[str, list[dict]] = {kind: [] for kind in kinds}
    for (tr_u, tr_i, tr_r), (te_u, te_i, te_r) in folds:
        tr_rows = np.fromiter((uid_to_idx[u] for u in tr_u), dtype=np.int64, count=len(tr_u))
        tr_cols = np.fromiter((iid_to_idx[i] for i in tr_i), dtype=np.int64, count=len(tr_i))
        train = sp.csr_matrix((tr_r, (tr_rows, tr_cols)), shape=matrix.shape)
        te_rows = np.fromiter((uid_to_idx[u] for u in te_u), dtype=np.int64, count=len(te_u))
        te_cols = np.fromiter((iid_to_idx[i] for i in te_i), dtype=np.int64, count=len(te_i))
        engines = {
            kind: build_engine(
                kind, train, seed=seed, factors=factors, iterations=iterations
            )
            for kind in kinds
        }
        for kind, engine in engines.items():
            predicted = np.fromiter(
                (engine.predict(int(u), int(i)) for u, i in zip(te_rows, te_cols)),
                dtype=float,
                count=len(te_rows),
            )
            per_fold[kind].append(rating_metrics(te_r, predicted))
    out: dict[str, dict] = {}
    for kind, fold_metrics in per_fold.items():
        rmse = np.asarray([m["rmse"] for m in fold_metrics])
        mae = np.asarray([m["mae"] for m in fold_metrics])
        out[kind] = {
            "rmse": round(float(rmse.mean()), 4),
            "rmse_std": round(float(rmse.std()), 4),
            "mae": round(float(mae.mean()), 4),
            "mae_std": round(float(mae.std()), 4),
            "per_fold": fold_metrics,
        }
    return out


def cv_rating_eval(data_dir: str | Path = "data", **kwargs) -> dict:
    """``cv_rating_eval_from_arrays`` over the real ml-100k data."""
    dataset_dir = fetch_movielens(data_dir)
    users, items, ratings = load_ratings(dataset_dir)
    return cv_rating_eval_from_arrays(users, items, ratings, **kwargs)


def mean_metrics(
    ranked_by_user: dict[int, list[int]],
    test_items: dict[int, int],
    ks: tuple[int, ...] = KS,
) -> dict:
    """Aggregate ranking metrics across users.

    ``ranked_by_user`` maps user_id to an ordered list of raw item ids;
    ``test_items`` maps user_id to the (single) held-out item id.

    Reports HR@k, Recall@k, Precision@k, NDCG@k, MAP@k and MRR. With a single
    relevant item per user, Recall@k == HR@k by construction.
    """
    hits = {k: 0 for k in ks}
    recall = {k: 0 for k in ks}
    precision = {k: 0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    ap = {k: 0.0 for k in ks}
    reciprocal_rank = 0.0
    n = 0
    for user_id, target in test_items.items():
        ranked = ranked_by_user.get(user_id) or []
        if not ranked:
            continue
        n += 1
        position = {item_id: rank for rank, item_id in enumerate(ranked, start=1)}
        rank = position.get(target)
        if rank is None:
            continue
        reciprocal_rank += 1.0 / rank
        for k in ks:
            if rank <= k:
                hits[k] += 1
                recall[k] += 1  # single relevant item
                precision[k] += 1.0 / k
                ndcg[k] += 1.0 / np.log2(rank + 1)
                ap[k] += 1.0 / rank
    return {
        "n_users": n,
        "hr": {str(k): round(hits[k] / n, 4) for k in ks},
        "recall": {str(k): round(recall[k] / n, 4) for k in ks},
        "precision": {str(k): round(precision[k] / n, 4) for k in ks},
        "ndcg": {str(k): round(ndcg[k] / n, 4) for k in ks},
        "map": {str(k): round(ap[k] / n, 4) for k in ks},
        "mrr": round(reciprocal_rank / n, 4),
    }


def cf_baseline(
    state: dict,
    test_items: dict[int, int],
    *,
    kind: str = "als",
    ks: tuple[int, ...] = KS,
    factors: int = 64,
    iterations: int = 20,
) -> dict:
    """Top-``max(ks)`` items from a raw CF engine, no agent involved.

    ``kind`` selects the engine: ``als``, ``user`` or ``item``. If it matches
    the engine the state was trained with, the persisted model is reused;
    otherwise the engine is fitted on demand from the state matrix.
    """
    from recagent.cf import CF_KINDS, build_cf
    from recagent.model import Recommender

    if kind not in CF_KINDS:
        raise ValueError(f"kind must be one of {CF_KINDS}, got {kind!r}")
    matrix = state["matrix"]
    uid_to_idx, item_ids = state["uid_to_idx"], state["item_ids"]
    if kind == state.get("cf_kind", "als"):
        model = state["model"]
    elif kind == "als":
        model = Recommender(factors=factors, iterations=iterations).fit(matrix)
    else:
        model = build_cf(kind, matrix)
    ranked: dict[int, list[int]] = {}
    for user_id in test_items:
        if user_id not in uid_to_idx:
            continue
        top = model.recommend(matrix, uid_to_idx[user_id], n=max(ks))
        ranked[user_id] = [int(item_ids[idx]) for idx, _ in top]
    metrics = mean_metrics(ranked, test_items, ks)
    metrics["kind"] = kind
    return metrics


async def _agent_ids(agent: RecAgent, request: str, deps: ToolRegistry, retries: int = 2) -> list[int]:
    """Run the agent for one user, retrying failed runs (LLMs are nondeterministic)."""
    for attempt in range(retries + 1):
        try:
            result = await agent.arun(request, deps)
            output = result.output
            if output is not None and output.items:
                return [item.item_id for item in output.items]
        except Exception:  # noqa: BLE001, S112 — one bad run must not sink the eval
            continue
    return []


async def agent_baseline(
    agent: RecAgent,
    deps: ToolRegistry,
    test_items: dict[int, int],
    *,
    k: int,
    concurrency: int = 8,
) -> tuple[dict, list[tuple[int, int, list[int]]]]:
    """Run the agent per user; return aggregate metrics plus per-user details."""
    semaphore = asyncio.Semaphore(concurrency)

    async def one(user_id: int) -> tuple[int, list[int]]:
        async with semaphore:
            ids = await _agent_ids(agent, f"Recommend {k} items for user_id: {user_id}.", deps)
            return user_id, ids

    pairs = await asyncio.gather(*(one(u) for u in test_items))
    ranked = {user_id: ids for user_id, ids in pairs}
    details = [
        (user_id, test_items[user_id], ids)
        for user_id, ids in pairs
        if user_id in test_items
    ]
    return mean_metrics(ranked, test_items, ks=(1, 3, k)), details


def build_test_items(state: dict, data_dir: str, *, min_interactions: int = 5, seed: int = 42) -> dict[int, int]:
    """Re-derive the exact leave-one-out split used at training time."""
    dataset_dir = fetch_movielens(data_dir)
    users, items, ratings = load_ratings(dataset_dir)
    _, (test_users, test_items) = leave_one_out(
        users, items, ratings, min_interactions=min_interactions, seed=seed
    )
    known = state["uid_to_idx"]
    return {int(u): int(i) for u, i in zip(test_users, test_items) if int(u) in known}


def genre_share(state: dict, ranked: dict[int, list[int]]) -> dict[str, float]:
    """Share of returned items whose genres include each genre."""
    meta = state["items_meta"]
    counts: dict[str, int] = {}
    total = 0
    for ids in ranked.values():
        for item_id in ids:
            total += 1
            for g in meta.get(item_id, {}).get("genres", []):
                counts[g] = counts.get(g, 0) + 1
    return {g: round(c / total, 4) for g, c in sorted(counts.items(), key=lambda x: -x[1])} if total else {}


async def constraint_eval(
    agent: RecAgent,
    deps: ToolRegistry,
    users: list[int],
    *,
    constraint: str,
    k: int,
    concurrency: int = 8,
) -> tuple[dict[int, list[int]], list[tuple[int, list[int]]]]:
    """Agent with a hard natural-language constraint, plus the CF lists for comparison.

    The metric of interest is constraint compliance (e.g. genre precision) —
    something the pure CF engine cannot express at all.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def one(user_id: int) -> tuple[int, list[int]]:
        async with semaphore:
            request = (
                f"Recommend {k} items for user_id: {user_id}. "
                f"Constraint: every item must be {constraint}."
            )
            ids = await _agent_ids(agent, request, deps)
            return user_id, ids

    pairs = await asyncio.gather(*(one(u) for u in users))
    return dict(pairs), pairs


def cf_lists(state: dict, users: list[int], n: int) -> dict[int, list[int]]:
    model, matrix = state["model"], state["matrix"]
    uid_to_idx, item_ids = state["uid_to_idx"], state["item_ids"]
    return {
        user_id: [int(item_ids[idx]) for idx, _ in model.recommend(matrix, uid_to_idx[user_id], n=n)]
        for user_id in users
        if user_id in uid_to_idx
    }


def save_report(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
