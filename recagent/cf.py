"""Memory-based collaborative filtering, implemented from scratch in numpy.

Classic neighbourhood methods over the explicit-rating matrix:

- ``UserBasedCF`` — Pearson-correlated nearest users; predict a user's rating
  of an item as their mean plus the similarity-weighted average of neighbour
  deviations.
- ``ItemBasedCF`` — adjusted-cosine item similarity; predict from the user's
  own ratings of similar items.

Both expose the same interface as the ALS ``Recommender``
(``fit`` / ``recommend(matrix, user_idx, n)``) so the rest of the pipeline —
tool registry, CLI, eval, agent — treats them identically.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

CF_KINDS = ("als", "user", "item")


def build_cf(
    kind: str, matrix: sp.csr_matrix, min_sim: float = 0.0, topk: int | None = None
) -> UserBasedCF | ItemBasedCF:
    """Fit the memory-based engine for ``kind`` (``user`` or ``item``).

    ``als`` is deliberately not handled here — it lives in :mod:`recagent.model`
    and is selected via ``cf="als"`` instead. ``topk`` keeps only the top-k
    strongest similarities per row (sparse neighbourhood); ``None`` keeps the
    full dense similarity, which is only tractable at small scale.
    """
    kind = kind.lower()
    if kind not in ("user", "item"):
        raise ValueError(f"build_cf supports 'user'/'item', got {kind!r}")
    cls = UserBasedCF if kind == "user" else ItemBasedCF
    return cls(min_sim=min_sim, topk=topk).fit(matrix)


def _row_means(matrix: sp.csr_matrix) -> np.ndarray:
    """Mean rating per user over rated items only (0 for empty profiles)."""
    counts = matrix.getnnz(axis=1)
    sums = np.asarray(matrix.sum(axis=1)).ravel()
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)


def _col_means(matrix: sp.csr_matrix) -> np.ndarray:
    """Mean rating per item over raters only (0 for unrated items)."""
    counts = matrix.getnnz(axis=0)
    sums = np.asarray(matrix.sum(axis=0)).ravel()
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)


def _user_similarity(
    matrix: sp.csr_matrix, min_sim: float = 0.0, topk: int | None = None
) -> np.ndarray | sp.csr_matrix:
    """User-user Pearson similarity (cosine on mean-centered rows).

    ``topk`` truncates each row to its strongest positive entries and returns a
    sparse matrix — the only tractable form for large user counts (a dense
    user-user matrix is O(n_users^2) and useless beyond ~10k users).
    """
    matrix = matrix.tocsr()
    centered = matrix.copy()
    rows, cols = centered.nonzero()
    centered[rows, cols] -= _row_means(matrix)[rows]
    squared = centered.multiply(centered)
    norms = np.sqrt(np.asarray(squared.sum(axis=1)).ravel())
    inv = np.zeros_like(norms)
    np.divide(1.0, norms, out=inv, where=norms > 0)
    similarity = (sp.diags(inv) @ centered) @ (sp.diags(inv) @ centered).T
    if topk is None:
        similarity = similarity.toarray()
    if sp.issparse(similarity):
        similarity = _knn_truncate(_zero_diagonal(similarity), topk)
        similarity.data[similarity.data < min_sim] = 0.0
        similarity.eliminate_zeros()
    else:
        np.fill_diagonal(similarity, 0.0)
        similarity[similarity < min_sim] = 0.0
    return similarity


def _item_similarity(
    matrix: sp.csr_matrix, min_sim: float = 0.0, topk: int | None = None
) -> np.ndarray | sp.csr_matrix:
    """Item-item adjusted-cosine similarity (cosine on mean-centered columns).

    Item-item is naturally sparse — S has entries only for co-rated item pairs —
    so the ``topk`` sparse form scales to tens of thousands of items while the
    dense form stays for small matrices.
    """
    matrix = matrix.tocsr()
    centered = matrix.copy()
    rows, cols = centered.nonzero()
    centered[rows, cols] -= _col_means(matrix)[cols]
    squared = centered.multiply(centered)
    norms = np.sqrt(np.asarray(squared.sum(axis=0)).ravel())
    inv = np.zeros_like(norms)
    np.divide(1.0, norms, out=inv, where=norms > 0)
    normalized = centered @ sp.diags(inv)
    similarity = normalized.T @ normalized
    if topk is None:
        similarity = similarity.toarray()
    if sp.issparse(similarity):
        similarity = _knn_truncate(_zero_diagonal(similarity), topk)
        similarity.data[similarity.data < min_sim] = 0.0
        similarity.eliminate_zeros()
    else:
        np.fill_diagonal(similarity, 0.0)
        similarity[similarity < min_sim] = 0.0
    return similarity


def _zero_diagonal(similarity: sp.csr_matrix | sp.csc_matrix) -> sp.csr_matrix:
    """Drop the diagonal of a sparse similarity matrix (format-agnostic)."""
    coo = similarity.tocoo()
    off_diag = coo.row != coo.col
    return sp.csr_matrix(
        (coo.data[off_diag], (coo.row[off_diag], coo.col[off_diag])),
        shape=coo.shape,
    )


def _knn_truncate(similarity: sp.csr_matrix, topk: int) -> sp.csr_matrix:
    """Keep only the top-``topk`` strongest positive entries per row (sparse)."""
    similarity = similarity.tocsr()
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for r in range(similarity.shape[0]):
        row = similarity.getrow(r)
        if row.nnz == 0:
            continue
        data = row.data
        indices = row.indices
        if len(data) > topk:
            take = np.argpartition(-data, topk - 1)[:topk]
            data = data[take]
            indices = indices[take]
        keep = data > 0
        rows.extend([r] * int(keep.sum()))
        cols.extend(indices[keep].tolist())
        vals.extend(data[keep].tolist())
    return sp.csr_matrix((vals, (rows, cols)), shape=similarity.shape)


def _top_similar(row: np.ndarray, idx: int, n: int) -> list[tuple[int, float]]:
    """Top-n positive entries of a similarity row, excluding the self entry."""
    out: list[tuple[int, float]] = []
    for candidate in np.argsort(-row):
        if len(out) == n:
            break
        if candidate == idx or row[candidate] <= 0:
            continue
        out.append((int(candidate), float(row[candidate])))
    return out


class UserBasedCF:
    """User-user neighbourhood collaborative filtering with Pearson correlation.

    ``topk`` (default ``None``) switches from the full dense similarity to a
    sparse top-k neighbourhood. The dense form is only tractable for small user
    counts; at large scale always pass ``topk``.
    """

    def __init__(self, min_sim: float = 0.0, topk: int | None = None):
        self.min_sim = min_sim
        self.topk = topk
        self.matrix: sp.csr_matrix | None = None
        self.user_means: np.ndarray | None = None
        self.centered: sp.csr_matrix | None = None
        self.similarity: np.ndarray | sp.csr_matrix | None = None
        self.similarity_norm: np.ndarray | sp.csr_matrix | None = None
        self._item_sim: np.ndarray | None = None

    def fit(self, matrix: sp.csr_matrix) -> UserBasedCF:
        self.matrix = matrix.tocsr()
        self.user_means = _row_means(self.matrix)
        centered = self.matrix.copy()
        rows, cols = centered.nonzero()
        centered[rows, cols] -= self.user_means[rows]
        self.centered = centered
        self.similarity = _user_similarity(self.matrix, self.min_sim, self.topk)
        self.similarity_norm = self._normalized_similarity()
        return self

    def _normalized_similarity(self) -> np.ndarray | sp.csr_matrix:
        """Rows of the similarity matrix divided by their L1 (|sim|) weight."""
        row_sums = np.asarray(np.abs(self.similarity).sum(axis=1)).ravel()
        if sp.issparse(self.similarity):
            inv = np.divide(1.0, row_sums, out=np.zeros_like(row_sums), where=row_sums != 0)
            return self.similarity.multiply(inv[:, None])
        normalized = np.zeros_like(self.similarity)
        np.divide(self.similarity, row_sums[:, None], out=normalized, where=row_sums[:, None] != 0)
        return normalized

    def _row(self, idx: int) -> np.ndarray:
        """Dense 1-D similarity row (converts the sparse top-k row when needed)."""
        row = self.similarity[idx]
        if sp.issparse(row):
            return row.toarray().ravel()
        return np.asarray(row).ravel()

    def similar_users(self, user_idx: int, n: int = 10) -> list[tuple[int, float]]:
        """Nearest neighbours of a user — the engine's own user-user similarity."""
        return _top_similar(self._row(user_idx), user_idx, n)

    def similar_items(self, item_idx: int, n: int = 10) -> list[tuple[int, float]]:
        """Nearest neighbours of an item (adjusted-cosine, computed lazily)."""
        if self._item_sim is None:
            self._item_sim = _item_similarity(self.matrix, self.min_sim)
        return _top_similar(self._item_sim[item_idx], item_idx, n)

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predicted rating: user mean + similarity-weighted neighbour deviation."""
        deviations = self.centered[:, item_idx].toarray().ravel()
        similarity = self._row(user_idx)
        denom = float(np.abs(similarity).sum())
        if denom == 0.0:
            return float(self.user_means[user_idx])
        return float(self.user_means[user_idx] + (similarity @ deviations) / denom)

    def recommend(self, matrix: sp.csr_matrix, user_idx: int, n: int = 10) -> list[tuple[int, float]]:
        """Top-n unseen items by predicted rating, best first (ALS-compatible)."""
        similarity = self._row(user_idx)
        denom = float(np.abs(similarity).sum())
        if denom == 0.0:
            scores = np.full(matrix.shape[1], self.user_means[user_idx])
        else:
            scores = self.user_means[user_idx] + (similarity / denom) @ self.centered
        rated = set(matrix.indices[matrix.indptr[user_idx] : matrix.indptr[user_idx + 1]])
        out: list[tuple[int, float]] = []
        for item_idx in np.argsort(-scores):
            if len(out) == n:
                break
            if item_idx in rated:
                continue
            out.append((int(item_idx), float(scores[item_idx])))
        return out

    def score_all(self) -> np.ndarray:
        """Dense (n_users x n_items) predicted ratings for every user.

        Matches per-user ``predict``/``recommend`` scoring in one batched op.
        Dense output is O(n_users x n_items); at large scale prefer per-user
        ``recommend``.
        """
        result = self.user_means[:, None] + self.similarity_norm @ self.centered
        return result.toarray() if sp.issparse(result) else result

    def save(self, path: str | Any) -> UserBasedCF:
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        coo = self.centered.tocoo()
        matrix_coo = self.matrix.tocoo()
        payload: dict[str, Any] = {
            "min_sim": self.min_sim,
            "topk": -1 if self.topk is None else self.topk,
            "user_means": self.user_means,
            "c_data": coo.data,
            "c_row": coo.row,
            "c_col": coo.col,
            "c_shape": np.asarray(coo.shape),
            "m_data": matrix_coo.data,
            "m_row": matrix_coo.row,
            "m_col": matrix_coo.col,
            "m_shape": np.asarray(matrix_coo.shape),
        }
        if sp.issparse(self.similarity):
            sim_coo = self.similarity.tocoo()
            payload.update(
                sim_sparse=1,
                sim_data=sim_coo.data,
                sim_row=sim_coo.row,
                sim_col=sim_coo.col,
                sim_shape=np.asarray(sim_coo.shape),
            )
        else:
            payload.update(sim_sparse=0, similarity=self.similarity)
        np.savez(path, **payload)
        return self

    @classmethod
    def load(cls, path: str | Any) -> UserBasedCF:
        saved = np.load(path)
        topk = int(saved["topk"]) if "topk" in saved else None
        obj = cls(min_sim=float(saved["min_sim"]), topk=None if topk == -1 else topk)
        obj.user_means = saved["user_means"]
        obj.centered = sp.coo_matrix(
            (saved["c_data"], (saved["c_row"], saved["c_col"])),
            shape=tuple(saved["c_shape"]),
        ).tocsr()
        obj.matrix = sp.coo_matrix(
            (saved["m_data"], (saved["m_row"], saved["m_col"])),
            shape=tuple(saved["m_shape"]),
        ).tocsr()
        if "sim_sparse" in saved and int(saved["sim_sparse"]):
            obj.similarity = sp.coo_matrix(
                (saved["sim_data"], (saved["sim_row"], saved["sim_col"])),
                shape=tuple(saved["sim_shape"]),
            ).tocsr()
        else:
            obj.similarity = saved["similarity"]
        obj.similarity_norm = obj._normalized_similarity()
        return obj


class ItemBasedCF:
    """Item-item neighbourhood collaborative filtering with adjusted cosine.

    ``topk`` (default ``None``) switches from the full dense similarity to a
    sparse top-k neighbourhood; item-item is naturally sparse, so ``topk``
    scales to tens of thousands of items.
    """

    def __init__(self, min_sim: float = 0.0, topk: int | None = None):
        self.min_sim = min_sim
        self.topk = topk
        self.matrix: sp.csr_matrix | None = None
        self.user_means: np.ndarray | None = None
        self.item_means: np.ndarray | None = None
        self.centered: sp.csr_matrix | None = None
        self.similarity: np.ndarray | sp.csr_matrix | None = None
        self._user_sim: np.ndarray | None = None

    def fit(self, matrix: sp.csr_matrix) -> ItemBasedCF:
        self.matrix = matrix.tocsr()
        self.item_means = _col_means(self.matrix)
        centered = self.matrix.copy()
        rows, cols = centered.nonzero()
        centered[rows, cols] -= self.item_means[cols]
        self.centered = centered
        self.user_means = _row_means(self.matrix)
        self.similarity = _item_similarity(self.matrix, self.min_sim, self.topk)
        return self

    def _row(self, idx: int) -> np.ndarray:
        """Dense 1-D similarity row (converts the sparse top-k row when needed)."""
        row = self.similarity[idx]
        if sp.issparse(row):
            return row.toarray().ravel()
        return np.asarray(row).ravel()

    def similar_items(self, item_idx: int, n: int = 10) -> list[tuple[int, float]]:
        """Nearest neighbours of an item — the engine's own item-item similarity."""
        return _top_similar(self._row(item_idx), item_idx, n)

    def similar_users(self, user_idx: int, n: int = 10) -> list[tuple[int, float]]:
        """Nearest neighbours of a user (Pearson, computed lazily)."""
        if self._user_sim is None:
            self._user_sim = _user_similarity(self.matrix, self.min_sim)
        return _top_similar(self._user_sim[user_idx], user_idx, n)

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predicted rating: similarity-weighted mean of the user's own ratings."""
        if self.matrix is None or self.similarity is None:
            raise ValueError("fit() must be called before predict()")
        similarity = self._row(item_idx)
        row_start, row_end = self.matrix.indptr[user_idx : user_idx + 2]
        rated_items = self.matrix.indices[row_start:row_end]
        rated_values = self.matrix.data[row_start:row_end]
        weights = similarity[rated_items]
        numer = float(np.dot(weights, rated_values))
        denom = float(np.abs(weights).sum())
        if denom == 0.0:
            return float(self.user_means[user_idx])
        return numer / denom

    def _score(self, user_row: np.ndarray) -> np.ndarray:
        """Predicted ratings for every item given a user's raw rating row."""
        if self.similarity is None:
            raise ValueError("fit() must be called before scoring")
        rated_mask = user_row != 0
        numer = self.similarity @ user_row
        denom = np.abs(self.similarity) @ rated_mask.astype(float)
        if sp.issparse(numer):
            numer = numer.toarray().ravel()
        if sp.issparse(denom):
            denom = denom.toarray().ravel()
        preds = np.zeros(self.similarity.shape[0])
        np.divide(numer, denom, out=preds, where=denom != 0)
        return preds

    def recommend(self, matrix: sp.csr_matrix, user_idx: int, n: int = 10) -> list[tuple[int, float]]:
        row = np.asarray(matrix.getrow(user_idx).toarray()).ravel()
        scores = self._score(row)
        scores[scores == 0.0] = self.user_means[user_idx]
        rated = set(matrix.indices[matrix.indptr[user_idx] : matrix.indptr[user_idx + 1]])
        out: list[tuple[int, float]] = []
        for item_idx in np.argsort(-scores):
            if len(out) == n:
                break
            if item_idx in rated:
                continue
            out.append((int(item_idx), float(scores[item_idx])))
        return out

    def score_all(self) -> np.ndarray:
        """Dense (n_users x n_items) predicted ratings for every user.

        Dense output is O(n_users x n_items); at large scale prefer per-user
        ``recommend``.
        """
        rows = self.matrix.tocsr()
        numer = (self.similarity @ rows.T).T  # S @ r_u for each user
        denom = (np.abs(self.similarity) @ rows.sign().T).T
        if sp.issparse(numer):
            numer = numer.toarray()
        if sp.issparse(denom):
            denom = denom.toarray()
        preds = np.zeros_like(numer)
        np.divide(numer, denom, out=preds, where=denom != 0)
        preds[denom == 0] = np.broadcast_to(self.user_means[:, None], preds.shape)[denom == 0]
        return preds

    def save(self, path: str | Any) -> ItemBasedCF:
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        coo = self.matrix.tocoo()
        payload: dict[str, Any] = {
            "min_sim": self.min_sim,
            "topk": -1 if self.topk is None else self.topk,
            "item_means": self.item_means,
            "user_means": self.user_means,
            "m_data": coo.data,
            "m_row": coo.row,
            "m_col": coo.col,
            "m_shape": np.asarray(coo.shape),
        }
        if sp.issparse(self.similarity):
            sim_coo = self.similarity.tocoo()
            payload.update(
                sim_sparse=1,
                sim_data=sim_coo.data,
                sim_row=sim_coo.row,
                sim_col=sim_coo.col,
                sim_shape=np.asarray(sim_coo.shape),
            )
        else:
            payload.update(sim_sparse=0, similarity=self.similarity)
        np.savez(path, **payload)
        return self

    @classmethod
    def load(cls, path: str | Any) -> ItemBasedCF:
        saved = np.load(path)
        topk = int(saved["topk"]) if "topk" in saved else None
        obj = cls(min_sim=float(saved["min_sim"]), topk=None if topk == -1 else topk)
        obj.matrix = sp.coo_matrix(
            (saved["m_data"], (saved["m_row"], saved["m_col"])),
            shape=tuple(saved["m_shape"]),
        ).tocsr()
        obj.item_means = saved["item_means"]
        obj.user_means = saved["user_means"]
        if "sim_sparse" in saved and int(saved["sim_sparse"]):
            obj.similarity = sp.coo_matrix(
                (saved["sim_data"], (saved["sim_row"], saved["sim_col"])),
                shape=tuple(saved["sim_shape"]),
            ).tocsr()
        else:
            obj.similarity = saved["similarity"]
        return obj
