from recagent.cli import build_parser


def test_train_cf_flag_defaults_to_user():
    args = build_parser().parse_args(["train", "--data", "data"])
    assert args.cf == "user"


def test_train_cf_flag_accepts_all_engines():
    for kind in ("als", "user", "item"):
        args = build_parser().parse_args(["train", "--cf", kind])
        assert args.cf == kind
