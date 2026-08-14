"""Persistence for trained artefacts (model + index spaces + metadata)."""

from __future__ import annotations

import pickle
from pathlib import Path


def save_state(state: dict, artifacts_dir: str | Path = "artifacts") -> None:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    state["model"].save(artifacts_dir / "model.npz")
    persisted = {k: v for k, v in state.items() if k != "model"}
    persisted.setdefault("cf_kind", "als")
    with open(artifacts_dir / "state.pkl", "wb") as f:
        pickle.dump(persisted, f)


def load_state(artifacts_dir: str | Path = "artifacts") -> dict:
    from recagent.cf import ItemBasedCF, UserBasedCF
    from recagent.model import Recommender

    artifacts_dir = Path(artifacts_dir)
    with open(artifacts_dir / "state.pkl", "rb") as f:
        state = pickle.load(f)
    model_path = artifacts_dir / "model.npz"
    kind = state.get("cf_kind", "als")
    if kind == "als":
        state["model"] = Recommender.load(model_path)
    elif kind == "user":
        state["model"] = UserBasedCF.load(model_path)
    elif kind == "item":
        state["model"] = ItemBasedCF.load(model_path)
    else:
        raise ValueError(f"unknown cf_kind {kind!r}; expected one of {('als', 'user', 'item')}")
    return state
