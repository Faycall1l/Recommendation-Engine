"""Persistence for trained artefacts (model + index spaces + metadata)."""

from __future__ import annotations

import pickle
from pathlib import Path


def save_state(state: dict, artifacts_dir: str | Path = "artifacts") -> None:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    state["model"].save(artifacts_dir / "model.npz")
    with open(artifacts_dir / "state.pkl", "wb") as f:
        pickle.dump({k: v for k, v in state.items() if k != "model"}, f)


def load_state(artifacts_dir: str | Path = "artifacts") -> dict:
    from recagent.model import Recommender

    artifacts_dir = Path(artifacts_dir)
    with open(artifacts_dir / "state.pkl", "rb") as f:
        state = pickle.load(f)
    state["model"] = Recommender.load(artifacts_dir / "model.npz")
    return state
