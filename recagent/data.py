"""Dataset loading and preprocessing for MovieLens 100K."""

from __future__ import annotations

import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ML_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

GENRES = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def fetch_movielens(root: str | Path = "data") -> Path:
    """Download and extract ml-100k if missing; return the dataset directory."""
    root = Path(root)
    dataset_dir = root / "ml-100k"
    if dataset_dir.exists():
        return dataset_dir
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "ml-100k.zip"
    if not archive.exists():
        urllib.request.urlretrieve(ML_100K_URL, archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(root)
    return dataset_dir


def load_ratings(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse u.data into (user_ids, item_ids, ratings) arrays."""
    users, items, ratings = [], [], []
    with open(dataset_dir / "u.data") as f:
        for line in f:
            u, i, r, *_ = line.rstrip("\n").split("\t")
            users.append(int(u))
            items.append(int(i))
            ratings.append(float(r))
    return np.asarray(users), np.asarray(items), np.asarray(ratings)


def load_items(dataset_dir: Path) -> dict[int, dict]:
    """Parse u.item into {item_id: {title, genres}}."""
    items = {}
    with open(dataset_dir / "u.item", encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 24:
                continue
            genres = [g for g, flag in zip(GENRES, parts[5:24]) if flag == "1"]
            items[int(parts[0])] = {"title": parts[1].strip(), "genres": genres}
    return items


def encode(
    users: np.ndarray,
    items: np.ndarray,
    ratings: np.ndarray,
    *,
    user_ids: np.ndarray | None = None,
    item_ids: np.ndarray | None = None,
) -> tuple[sp.csr_matrix, dict[int, int], dict[int, int], np.ndarray, np.ndarray]:
    """Map raw ids to contiguous indices and build a sparse user-item matrix.

    The index spaces can be pinned via ``user_ids``/``item_ids`` so that
    train/test splits share one consistent space.
    """
    user_ids = user_ids if user_ids is not None else np.unique(users)
    item_ids = item_ids if item_ids is not None else np.unique(items)
    uid_to_idx = {u: i for i, u in enumerate(user_ids)}
    iid_to_idx = {i: j for j, i in enumerate(item_ids)}
    rows = np.fromiter((uid_to_idx[u] for u in users), dtype=np.int32, count=len(users))
    cols = np.fromiter((iid_to_idx[i] for i in items), dtype=np.int32, count=len(items))
    matrix = sp.csr_matrix(
        (ratings, (rows, cols)), shape=(len(user_ids), len(item_ids))
    )
    return matrix, uid_to_idx, iid_to_idx, user_ids, item_ids


def leave_one_out(
    users: np.ndarray,
    items: np.ndarray,
    ratings: np.ndarray,
    *,
    min_interactions: int = 5,
    seed: int = 42,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Hold out one interaction per user with enough history.

    Returns ``((train_users, train_items, train_ratings), (test_users, test_items))``.
    """
    rng = np.random.default_rng(seed)
    counts = Counter(users)
    eligible = [u for u, c in counts.items() if c >= min_interactions]
    held_idx = set()
    for u in eligible:
        candidates = np.flatnonzero(users == u)
        held_idx.add(int(rng.choice(candidates)))
    held = np.fromiter(sorted(held_idx), dtype=np.int64)
    mask = np.ones(len(users), dtype=bool)
    mask[held] = False
    return (users[mask], items[mask], ratings[mask]), (users[held], items[held])
