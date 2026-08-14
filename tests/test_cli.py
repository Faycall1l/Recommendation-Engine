import scipy.sparse as sp

from recagent.cf import UserBasedCF
from recagent.cli import build_parser, main
from recagent.state import save_state


def _state(tmp_path):
    matrix = sp.csr_matrix([[5.0, 3.0, 0.0, 0.0], [0.0, 1.0, 3.0, 0.0], [2.0, 0.0, 4.0, 0.0]])
    state = {
        "model": UserBasedCF().fit(matrix),
        "matrix": matrix,
        "uid_to_idx": {1: 0, 2: 1, 3: 2},
        "iid_to_idx": {11: 0, 12: 1, 13: 2, 14: 3},
        "user_ids": [1, 2, 3],
        "item_ids": [11, 12, 13, 14],
        "items_meta": {13: {"title": "Twin Peaks"}},
        "cf_kind": "user",
    }
    save_state(state, tmp_path)
    return tmp_path


def test_train_cf_flag_defaults_to_user():
    args = build_parser().parse_args(["train", "--data", "data"])
    assert args.cf == "user"


def test_train_cf_flag_accepts_all_engines():
    for kind in ("als", "user", "item"):
        args = build_parser().parse_args(["train", "--cf", kind])
        assert args.cf == kind


def test_data_kind_flags():
    args = build_parser().parse_args(["train"])
    assert args.data_kind == "ml-100k"
    args = build_parser().parse_args(["train", "--data-kind", "ml-20m"])
    assert args.data_kind == "ml-20m"
    args = build_parser().parse_args(["eval", "--data-kind", "ml-20m", "--sample-ratings", "5000"])
    assert args.data_kind == "ml-20m"
    assert args.sample_ratings == 5000
    args = build_parser().parse_args(["eval"])
    assert args.sample_ratings is None


def test_recommend_prints_engine_kind(tmp_path, capsys):
    main(["recommend", "1", "--artifacts", str(_state(tmp_path))])
    out = capsys.readouterr().out
    assert "user-based CF" in out
    assert "Twin Peaks" in out


def test_explain_prints_evidence(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("recagent.config.load_llm_config", _disabled_config)
    main(["explain", "1", "13", "--artifacts", str(_state(tmp_path))])
    out = capsys.readouterr().out
    assert "basis:" in out
    assert "LLM disabled" in out


def _disabled_config():
    from recagent.config import LLMConfig

    return LLMConfig(enabled=False)


def test_eval_cf_flags():
    args = build_parser().parse_args(["eval", "--cf", "item"])
    assert args.cf == "item"
    assert not args.all_cf
    args = build_parser().parse_args(["eval", "--all-cf"])
    assert args.all_cf
    assert args.cf is None
    args = build_parser().parse_args(["eval"])
    assert args.cf is None
    assert not args.all_cf


def test_eval_protocol_flags():
    args = build_parser().parse_args(["eval"])
    assert args.protocol == "ranking"
    assert args.folds == 5
    assert not args.baselines
    assert not args.bootstrap
    args = build_parser().parse_args(
        ["eval", "--protocol", "rating", "--folds", "3", "--baselines", "--bootstrap"]
    )
    assert args.protocol == "rating"
    assert args.folds == 3
    assert args.baselines
    assert args.bootstrap
    args = build_parser().parse_args(["eval", "--protocol", "all"])
    assert args.protocol == "all"


def test_eval_exclude_head_flag():
    args = build_parser().parse_args(["eval", "--exclude-head", "0.02"])
    assert args.exclude_head == 0.02
    args = build_parser().parse_args(["eval"])
    assert args.exclude_head is None


def test_train_ml20m_end_to_end(tmp_path, monkeypatch, capsys):
    from recagent.data import load_items_20m, load_ratings_20m

    # synthetic ml-20m layout: 3 users x 6 ratings, enough for min_interactions=5
    dataset = tmp_path / "ml-20m"
    dataset.mkdir()
    rows = []
    for u in range(1, 4):
        for j in range(6):
            rows.append(f"{u},{100 + u * 10 + j},{float(1 + (u + j) % 5)},12345\n")
    (dataset / "ratings.csv").write_text("userId,movieId,rating,timestamp\n" + "".join(rows))
    (dataset / "movies.csv").write_text(
        "movieId,title,genres\n101,Toy Story (1995),Comedy\n102,Seven Samurai (1954),Drama\n"
    )

    monkeypatch.setattr(
        "recagent.data.loaders",
        lambda kind: (lambda d: dataset, load_ratings_20m, load_items_20m),
    )
    artifacts = tmp_path / "art"
    main(["train", "--data", str(tmp_path), "--data-kind", "ml-20m", "--artifacts", str(artifacts)])
    out = capsys.readouterr().out
    assert "saved artefacts" in out
    from recagent.state import load_state

    state = load_state(artifacts)
    assert state["cf_kind"] == "user"
    assert state["matrix"].shape == (3, 15)


def test_eval_rating_protocol_writes_report(tmp_path, capsys, monkeypatch):
    import json

    fake = {"mf": {"rmse": 0.95, "rmse_std": 0.02, "mae": 0.7, "mae_std": 0.01, "per_fold": []}}
    monkeypatch.setattr("recagent.evaluate.cv_rating_eval", lambda *a, **k: fake)
    main(
        [
            "eval",
            "--protocol",
            "rating",
            "--data",
            str(tmp_path),
            "--artifacts",
            str(tmp_path),
            "--rating-report",
            str(tmp_path / "eval_rating.json"),
            "--report",
            str(tmp_path / "eval_report.json"),
        ]
    )
    out = capsys.readouterr().out
    assert "5-fold CV explicit-rating prediction" in out
    report = json.loads((tmp_path / "eval_rating.json").read_text())
    assert report["protocol"] == "rating"
    assert report["engines"]["mf"]["rmse"] == 0.95
