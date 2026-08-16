"""Persistence for trained artefacts (model + index spaces + metadata)."""

from __future__ import annotations

import pickle
from pathlib import Path


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only allows safe builtins + numpy/scipy types.

    Blocks arbitrary code execution via crafted pickle payloads.
    """

    _ALLOWED_MODULES = frozenset({"numpy", "scipy"})

    def find_class(self, module: str, name: str) -> type:
        for prefix in self._ALLOWED_MODULES:
            if module.startswith(prefix):
                import importlib

                mod = importlib.import_module(module)
                obj = getattr(mod, name, None)
                if obj is not None:
                    return obj
                raise pickle.UnpicklingError(
                    f"{module}.{name} not in whitelist"
                )
        raise pickle.UnpicklingError(
            f"class {module}.{name} is not allowed by RestrictedUnpickler"
        )


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
        state = RestrictedUnpickler(f).load()
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
