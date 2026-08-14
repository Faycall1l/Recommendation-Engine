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


def test_recommend_prints_engine_kind(tmp_path, capsys):
    main(["recommend", "1", "--artifacts", str(_state(tmp_path))])
    out = capsys.readouterr().out
    assert "user-based CF" in out
    assert "Twin Peaks" in out
