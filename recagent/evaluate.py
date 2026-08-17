"""Offline evaluation: does the agent beat the raw collaborative filter?

Leave-one-out holdout (same split as training) scored with hit-rate and NDCG.
The agent is scored on the exact lists it emits, so a better result is evidence
the reasoning+tools stage adds signal beyond ALS alone.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from recagent.agent import RecAgent
from recagent.data import encode, leave_one_out, loaders, split_ratings
from recagent.tools import ToolRegistry

KS = (1, 3, 5, 10)


@dataclasses.dataclass
class ConstraintResult:
    """Result of a constraint evaluation run."""

    constraint: str
    agent_lists: dict[int, list[int]]
    cf_lists: dict[int, list[int]]
    users: list[int]
    compliance_rate: float = 0.0
    agent_genre_precision: dict[str, float] = dataclasses.field(default_factory=dict)
    cf_genre_precision: dict[str, float] = dataclasses.field(default_factory=dict)


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
    regularization: float = 0.1,
    sample_ratings: int | None = None,
    verbose: bool = False,
    engine_kwargs: dict[str, dict] | None = None,
) -> dict:
    """5-fold CV explicit-rating prediction: RMSE/MAE mean+-std per engine.

    Engines are refit from scratch on each fold's train matrix and scored on
    the fold's held-out triples. ``als`` is rejected (no calibrated predict).
    ``sample_ratings`` deterministically subsamples rating triples before the
    fold split — the standard way to make CV tractable on 20M-scale data.
    ``verbose`` prints a progress line per fold (long runs are otherwise silent).
    ``engine_kwargs`` maps a kind to per-engine fit kwargs (e.g.
    ``{"mf": {"factors": 6, "iterations": 15, "regularization": 1.0}}``)
    layered over the shared defaults — unit-weight ALS needs far stronger
    regularization and fewer factors than the implicit engines.
    """
    from recagent.engines import RATING_ENGINES, build_engine

    kinds = list(kinds or RATING_ENGINES)
    for kind in kinds:
        if kind not in RATING_ENGINES:
            raise ValueError(f"rating protocol supports {RATING_ENGINES}, got {kind!r}")
    engine_kwargs = engine_kwargs or {}
    if sample_ratings is not None:
        rng = np.random.default_rng(seed)
        take = rng.choice(len(ratings), size=sample_ratings, replace=False)
        users, items, ratings = users[take], items[take], ratings[take]
    matrix, uid_to_idx, iid_to_idx, _user_ids, _item_ids = encode(users, items, ratings)
    folds = split_ratings(users, items, ratings, k=k, seed=seed)
    per_fold: dict[str, list[dict]] = {kind: [] for kind in kinds}
    for fold_idx, ((tr_u, tr_i, tr_r), (te_u, te_i, te_r)) in enumerate(folds):
        if verbose:
            print(
                f"  fold {fold_idx + 1}/{k}: {len(tr_r):,} train, {len(te_r):,} test triples",
                flush=True,
            )
        tr_rows = np.fromiter((uid_to_idx[u] for u in tr_u), dtype=np.int64, count=len(tr_u))
        tr_cols = np.fromiter((iid_to_idx[i] for i in tr_i), dtype=np.int64, count=len(tr_i))
        train = sp.csr_matrix((tr_r, (tr_rows, tr_cols)), shape=matrix.shape)
        te_rows = np.fromiter((uid_to_idx[u] for u in te_u), dtype=np.int64, count=len(te_u))
        te_cols = np.fromiter((iid_to_idx[i] for i in te_i), dtype=np.int64, count=len(te_i))
        engines = {}
        for kind in kinds:
            kwargs = {"seed": seed, "factors": factors, "iterations": iterations,
                      "regularization": regularization}
            kwargs.update(engine_kwargs.get(kind, {}))
            engines[kind] = build_engine(kind, train, **kwargs)
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
        config = {"factors": factors, "iterations": iterations, "regularization": regularization}
        config.update(engine_kwargs.get(kind, {}))
        out[kind] = {
            "rmse": round(float(rmse.mean()), 4),
            "rmse_std": round(float(rmse.std()), 4),
            "mae": round(float(mae.mean()), 4),
            "mae_std": round(float(mae.std()), 4),
            "config": config,
            "per_fold": fold_metrics,
        }
    return out


def cv_rating_eval(data_dir: str | Path = "data", *, data_kind: str = "ml-100k", **kwargs) -> dict:
    """``cv_rating_eval_from_arrays`` over a real MovieLens dataset."""
    fetch, load_ratings_fn, _load_items = loaders(data_kind)
    dataset_dir = fetch(data_dir)
    users, items, ratings = load_ratings_fn(dataset_dir)
    return cv_rating_eval_from_arrays(users, items, ratings, **kwargs)


def head_item_ids(items: np.ndarray, fraction: float) -> set[int]:
    """The ``fraction`` most-popular raw item ids (by rating count, ties by id)."""
    counts = Counter(items)
    ordered = sorted(counts, key=lambda item_id: (-counts[item_id], item_id))
    n_head = round(fraction * len(ordered))
    return {int(item_id) for item_id in ordered[:n_head]}


def loo_ranking_eval_from_arrays(
    users: np.ndarray,
    items: np.ndarray,
    ratings: np.ndarray,
    *,
    kinds: Sequence[str] | None = None,
    min_interactions: int = 5,
    seed: int = 42,
    factors: int = 64,
    iterations: int = 20,
    ks: tuple[int, ...] = KS,
    user_sample: int | None = None,
    engine_kwargs: dict[str, dict] | None = None,
    exclude_head: float | None = None,
) -> dict:
    """Leave-one-out ranking eval across any ranking engines.

    One held-out interaction per user (same split as training), scored with
    the full metric set from :func:`mean_metrics`. ``engine_kwargs`` maps a
    kind to per-engine fit kwargs (e.g. ``{"mf": {...}}``) layered over the
    shared defaults. ``exclude_head`` (0 <= f < 1) drops test targets that are
    among the ``f`` most-popular items — the Cremonesi–Koren–Turrin (2010)
    debias that stops raw popularity from dominating the protocol. Returns a
    dict keyed by engine kind.
    """
    from recagent.engines import RANKING_ENGINES, build_engine

    kinds = list(kinds or RANKING_ENGINES)
    for kind in kinds:
        if kind not in RANKING_ENGINES:
            raise ValueError(f"ranking protocol supports {RANKING_ENGINES}, got {kind!r}")
    if exclude_head is not None and not 0.0 <= exclude_head < 1.0:
        raise ValueError(f"exclude_head must be in [0, 1), got {exclude_head!r}")
    (tr_u, tr_i, tr_r), (te_u, te_i) = leave_one_out(
        users, items, ratings, min_interactions=min_interactions, seed=seed
    )
    matrix, uid_to_idx, _iid_to_idx, _user_ids, item_ids = encode(tr_u, tr_i, tr_r)
    engine_kwargs = engine_kwargs or {}
    engines = {}
    for kind in kinds:
        kwargs = {"seed": seed, "factors": factors, "iterations": iterations}
        kwargs.update(engine_kwargs.get(kind, {}))
        engines[kind] = build_engine(kind, matrix, **kwargs)
    test_items = {int(u): int(i) for u, i in zip(te_u, te_i) if int(u) in uid_to_idx}
    if exclude_head is not None:
        head = head_item_ids(items, exclude_head)
        test_items = {u: i for u, i in test_items.items() if i not in head}
    if user_sample is not None:
        test_items = {u: test_items[u] for u in sorted(test_items)[:user_sample]}
    ranked: dict[str, dict[int, list[int]]] = {kind: {} for kind in kinds}
    per_user_rank: dict[str, dict[int, int]] = {kind: {} for kind in kinds}
    for user_id in test_items:
        user_idx = uid_to_idx[user_id]
        for kind, engine in engines.items():
            top = engine.recommend(matrix, user_idx, n=max(ks))
            ranked[kind][user_id] = [int(item_ids[idx]) for idx, _ in top]
            try:
                per_user_rank[kind][user_id] = ranked[kind][user_id].index(test_items[user_id]) + 1
            except ValueError:
                per_user_rank[kind][user_id] = 0
    out: dict[str, dict] = {}
    for kind in kinds:
        metrics = mean_metrics(ranked[kind], test_items, ks)
        metrics["kind"] = kind
        metrics["exclude_head"] = exclude_head
        metrics["per_user_rank"] = {str(u): r for u, r in per_user_rank[kind].items()}
        out[kind] = metrics
    return out


def loo_ranking_eval(data_dir: str | Path = "data", *, data_kind: str = "ml-100k", **kwargs) -> dict:
    """``loo_ranking_eval_from_arrays`` over a real MovieLens dataset."""
    fetch, load_ratings_fn, _load_items = loaders(data_kind)
    dataset_dir = fetch(data_dir)
    users, items, ratings = load_ratings_fn(dataset_dir)
    return loo_ranking_eval_from_arrays(users, items, ratings, **kwargs)


def hits_from_ranks(per_user_rank: dict[str, int] | dict[int, int], k: int) -> np.ndarray:
    """Per-user hit@k boolean vector from a {user: 1-based rank-or-0} map."""
    uids = sorted(per_user_rank)
    return np.asarray([1 if per_user_rank[u] and per_user_rank[u] <= k else 0 for u in uids])


def mrr_from_ranks(per_user_rank: dict[str, int] | dict[int, int]) -> np.ndarray:
    """Per-user reciprocal rank (0 when the target was not ranked) vector."""
    uids = sorted(per_user_rank)
    return np.asarray([1.0 / per_user_rank[u] if per_user_rank[u] else 0.0 for u in uids])


def aligned_rank_arrays(
    ranks_a: dict[str, int] | dict[int, int],
    ranks_b: dict[str, int] | dict[int, int],
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Hit@k and MRR vectors for two {user: rank} maps, paired by user.

    Ranks maps may mix int/str user ids (CF baselines save str keys, agent
    runs build int keys); pairing by *position* after sorting each map
    independently would misalign users. This normalizes both to int keys and
    pairs on the sorted common users, returning (hit_a, hit_b, mrr_a, mrr_b,
    uids).
    """
    a = {int(u): r for u, r in ranks_a.items()}
    b = {int(u): r for u, r in ranks_b.items()}
    uids = sorted(a.keys() & b.keys())
    hit_a = hits_from_ranks({u: a[u] for u in uids}, k)
    hit_b = hits_from_ranks({u: b[u] for u in uids}, k)
    mrr_a = mrr_from_ranks({u: a[u] for u in uids})
    mrr_b = mrr_from_ranks({u: b[u] for u in uids})
    return hit_a, hit_b, mrr_a, mrr_b, uids


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """Paired bootstrap on the mean difference ``a - b``.

    Resamples user indices with replacement; reports the CI of the mean
    difference, a two-sided p-value for diff == 0 under the shifted null,
    and Cohen's d effect size ( pooled-SD normalised mean difference ).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired scores must be equal length")
    if len(a) == 0:
        raise ValueError("need at least one pair of scores")
    rng = np.random.default_rng(seed)
    n = len(a)
    diff = a - b
    mean_diff = float(diff.mean())
    # pooled standard deviation for Cohen's d
    pooled_std = float(np.sqrt((a.var() + b.var()) / 2.0))
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
    # vectorised bootstrap: generate all resampled means at once
    indices = rng.integers(0, n, size=(n_boot, n))
    boot = (diff[indices]).mean(axis=1)
    lo = (1.0 - ci) / 2.0
    ci_lo, ci_hi = float(np.quantile(boot, lo)), float(np.quantile(boot, 1.0 - lo))
    centered = boot - mean_diff
    p_value = float(np.mean(np.abs(centered) >= abs(mean_diff)))
    return {
        "mean_diff": round(mean_diff, 4),
        "ci_lo": round(ci_lo, 4),
        "ci_hi": round(ci_hi, 4),
        "ci_level": ci,
        "n_boot": n_boot,
        "p_value": round(p_value, 4),
        "cohens_d": round(cohens_d, 4),
        "n": n,
    }


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
    per_user_rank: dict[int, int] = {}
    for user_id, target in test_items.items():
        if user_id not in uid_to_idx:
            continue
        top = model.recommend(matrix, uid_to_idx[user_id], n=max(ks))
        ranked[user_id] = [int(item_ids[idx]) for idx, _ in top]
        try:
            per_user_rank[user_id] = ranked[user_id].index(target) + 1
        except ValueError:
            per_user_rank[user_id] = 0
    metrics = mean_metrics(ranked, test_items, ks)
    metrics["kind"] = kind
    metrics["per_user_rank"] = {str(u): r for u, r in per_user_rank.items()}
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


def build_test_items(
    state: dict,
    data_dir: str,
    *,
    min_interactions: int = 5,
    seed: int = 42,
    data_kind: str = "ml-100k",
    exclude_head: float | None = None,
    cache_dir: str | None = None,
) -> dict[int, int]:
    """Re-derive the exact leave-one-out split used at training time.

    ``exclude_head`` (0 <= f < 1) drops test targets among the ``f``
    most-popular items, leaving a long-tail cohort (Cremonesi–Koren–Turrin).

    When ``cache_dir`` is provided the computed split is persisted to disk
    and reused on subsequent calls with the same parameters, avoiding the
    expensive re-derivation (especially on ml-20m where this takes ~9 min).
    """
    import hashlib
    import json
    from pathlib import Path

    # Build a deterministic cache key from the parameters
    cache_key = hashlib.sha256(
        f"{data_kind}:{seed}:{min_interactions}:{exclude_head}".encode()
    ).hexdigest()[:16]

    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"loo_split_{cache_key}.json"
        if cache_path.exists():
            raw = json.loads(cache_path.read_text())
            return {int(k): int(v) for k, v in raw.items()}

    fetch, load_ratings_fn, _load_items = loaders(data_kind)
    dataset_dir = fetch(data_dir)
    users, items, ratings = load_ratings_fn(dataset_dir)
    _, (test_users, test_items) = leave_one_out(
        users, items, ratings, min_interactions=min_interactions, seed=seed
    )
    if exclude_head is not None and not 0.0 <= exclude_head < 1.0:
        raise ValueError(f"exclude_head must be in [0, 1), got {exclude_head!r}")
    if exclude_head is not None:
        head = head_item_ids(items, exclude_head)
        pairs = [
            (int(u), int(i))
            for u, i in zip(test_users, test_items)
            if int(i) not in head
        ]
        test_users = np.asarray([u for u, _ in pairs], dtype=int)
        test_items = np.asarray([i for _, i in pairs], dtype=int)
    known = state["uid_to_idx"]
    result = {int(u): int(i) for u, i in zip(test_users, test_items) if int(u) in known}

    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        (cache_path / f"loo_split_{cache_key}.json").write_text(
            json.dumps({str(k): v for k, v in result.items()})
        )

    return result


def genre_precision(share: dict[str, float], genre: str) -> float:
    """Case-insensitive lookup in a genre-share dict (meta uses title-case)."""
    genre = genre.lower()
    for key, value in share.items():
        if key.lower() == genre:
            return value
    return 0.0


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


# ── Beyond-accuracy metrics ──────────────────────────────────────────


def intra_list_diversity(
    ranked_by_user: dict[int, list[int]],
    state: dict,
    k: int = 10,
) -> float:
    """Average intra-list diversity (1 − mean Jaccard over all user lists).

    For each user's top-k list, compute pairwise Jaccard distance between
    item genre sets, then average across all pairs and all users.
    Returns a value in [0, 1] — higher means more diverse lists.
    """
    meta = state["items_meta"]
    jaccards: list[float] = []
    for item_ids in ranked_by_user.values():
        top = item_ids[:k]
        if len(top) < 2:
            continue
        genre_sets = [set(meta.get(iid, {}).get("genres", [])) for iid in top]
        for a in range(len(genre_sets)):
            for b in range(a + 1, len(genre_sets)):
                union = genre_sets[a] | genre_sets[b]
                inter = genre_sets[a] & genre_sets[b]
                jaccards.append(1.0 - len(inter) / len(union) if union else 0.0)
    return round(float(np.mean(jaccards)), 4) if jaccards else 0.0


def average_novelty(
    ranked_by_user: dict[int, list[int]],
    state: dict,
    k: int = 10,
) -> float:
    """Average self-information novelty across all top-k lists.

    Novelty(item) = −log2(popularity), where popularity is the fraction of
    users who interacted with the item. Higher means the system surfaces
    less popular (more novel) items. Returns bits (log base 2).
    """
    matrix = state["matrix"]
    n_users = matrix.shape[0]
    total_pop = np.array(matrix.getnnz(axis=0)).flatten()
    popularities = total_pop / n_users
    # avoid log(0): clamp to 1/(2*n_users)
    popularities = np.clip(popularities, 1.0 / (2 * n_users), 1.0)
    neg_log_pop = -np.log2(popularities)
    item_ids = state["item_ids"]
    iid_to_neglog = {int(iid): float(neg_log_pop[idx]) for idx, iid in enumerate(item_ids)}
    novelties: list[float] = []
    for item_ids_list in ranked_by_user.values():
        for iid in item_ids_list[:k]:
            if iid in iid_to_neglog:
                novelties.append(iid_to_neglog[iid])
    return round(float(np.mean(novelties)), 4) if novelties else 0.0


def catalog_coverage(
    ranked_by_user: dict[int, list[int]],
    state: dict,
    k: int = 10,
) -> dict[str, float]:
    """Catalog coverage and Gini index of item exposure.

    Returns:
        coverage: fraction of all items that appear in at least one top-k list.
        gini: Gini coefficient over item exposure counts (0 = perfectly equal,
              1 = all exposure on one item).
    """
    item_ids = state["item_ids"]
    n_items = len(item_ids)
    exposure: dict[int, int] = {int(iid): 0 for iid in item_ids}
    for item_ids_list in ranked_by_user.values():
        for iid in item_ids_list[:k]:
            if iid in exposure:
                exposure[iid] += 1
    counts = np.array(list(exposure.values()), dtype=float)
    covered = int(np.sum(counts > 0))
    cov = round(covered / n_items, 4) if n_items else 0.0
    # Gini
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    index = np.arange(1, n + 1)
    gini = float(1.0 - 2.0 * np.sum(sorted_counts * (n - index + 0.5)) / (n * sorted_counts.sum())) if sorted_counts.sum() > 0 else 0.0
    return {"coverage": cov, "gini": round(gini, 4)}


def long_tail_share(
    ranked_by_user: dict[int, list[int]],
    state: dict,
    k: int = 10,
    head_fraction: float = 0.2,
) -> float:
    """Fraction of recommended items that are long-tail (not in the top ``head_fraction`` by popularity).

    Uses rating count as the popularity measure. Items outside the top
    ``head_fraction`` most-popular items are considered long-tail.
    """
    meta = state["items_meta"]
    counts: dict[int, int] = {}
    for item_ids_list in ranked_by_user.values():
        for iid in item_ids_list[:k]:
            counts[iid] = counts.get(iid, 0) + 1
    if not counts:
        return 0.0
    all_items = sorted(
        meta.keys(),
        key=lambda iid: -meta.get(iid, {}).get("rating_count", 0),
    )
    n_head = max(1, int(len(all_items) * head_fraction))
    head_ids = set(all_items[:n_head])
    n_long_tail = sum(1 for iid in counts if iid not in head_ids)
    return round(n_long_tail / len(counts), 4)


def serendipity(
    ranked_by_user: dict[int, list[int]],
    test_items: dict[int, int],
    state: dict,
    k: int = 10,
) -> float:
    """Average serendipity: items the user hadn't seen that are relevant but not obvious.

    Serendipity(item, user) = relevance(item) * (1 − popularity(item)) if
    the item was not in the user's profile, else 0. Returns the mean across
    all top-k lists.
    """
    matrix = state["matrix"]
    uid_to_idx = state["uid_to_idx"]
    item_ids = state["item_ids"]
    n_users = matrix.shape[0]
    total_pop = np.array(matrix.getnnz(axis=0)).flatten()
    popularities = total_pop / n_users
    iid_to_idx = {int(iid): idx for idx, iid in enumerate(item_ids)}
    scores_list: list[float] = []
    for user_id, rec_ids in ranked_by_user.items():
        if user_id not in uid_to_idx:
            continue
        user_idx = uid_to_idx[user_id]
        user_row = matrix.getrow(user_idx)
        rated = set(user_row.indices)
        for iid in rec_ids[:k]:
            if iid not in iid_to_idx:
                continue
            item_idx = iid_to_idx[iid]
            if item_idx in rated:
                continue
            pop = popularities[item_idx]
            # relevance proxy: CF score via matrix factorisation
            vec_u = np.asarray(user_row.todense()).flatten()
            vec_i = np.asarray(matrix.getcol(item_idx).todense()).flatten()
            sim = float(np.dot(vec_u, vec_i))
            scores_list.append(sim * (1.0 - pop))
    return round(float(np.mean(scores_list)), 4) if scores_list else 0.0


def demographic_parity(
    ranked_by_user: dict[int, list[int]],
    state: dict,
    k: int = 10,
) -> dict[str, float]:
    """Genre demographic parity: max share − min share across all genres.

    Measures whether the recommender disproportionately recommends items from
    certain genres. A lower value means more balanced genre representation.
    Returns {"max_share", "min_share", "disparity"} where
    disparity = max_share − min_share.
    """
    meta = state["items_meta"]
    genre_counts: dict[str, int] = {}
    total = 0
    for item_ids_list in ranked_by_user.values():
        for iid in item_ids_list[:k]:
            total += 1
            for g in meta.get(iid, {}).get("genres", []):
                genre_counts[g] = genre_counts.get(g, 0) + 1
    if not genre_counts or total == 0:
        return {"max_share": 0.0, "min_share": 0.0, "disparity": 0.0}
    shares = [c / total for c in genre_counts.values()]
    return {
        "max_share": round(max(shares), 4),
        "min_share": round(min(shares), 4),
        "disparity": round(max(shares) - min(shares), 4),
    }
