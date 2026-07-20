from __future__ import annotations

import json
from pathlib import Path

import pytest
from mantis_v2 import cli
from mantis_v2.rl_account import (
    RlAccountError,
    replay_account_fixture,
    write_account_replay_manifest,
)
from mantis_v2.rl_config import load_rl_config

ROOT = Path(__file__).resolve().parents[1]


def _config():
    return load_rl_config(ROOT / "configs" / "rl-entry-smoke.toml")


def _bar(timestamp: str, **values: object) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "action": "none",
        "contract": None,
        "quantity": 0,
        "side": None,
        "mark_ticks": "0",
        "pending_orders": 0,
        **values,
    }


def test_mll_starts_at_97k_ratchets_at_eod_only_and_locks_at_100k() -> None:
    fixture = {
        "schema_version": 1,
        "fixture_id": "mll-ratchet",
        "bars": [
            _bar("2026-07-20T19:00:00+00:00"),
            _bar("2026-07-20T20:10:00+00:00"),
            _bar("2026-07-21T19:00:00+00:00", realized_pnl="2000.00"),
            _bar("2026-07-21T20:10:00+00:00"),
            _bar("2026-07-22T19:00:00+00:00", realized_pnl="2000.00"),
            _bar("2026-07-22T20:10:00+00:00"),
            _bar("2026-07-23T19:00:00+00:00", realized_pnl="-1000.00"),
            _bar("2026-07-23T20:10:00+00:00"),
        ],
    }

    result = replay_account_fixture(_config(), fixture, ROOT.parent)

    assert [bar["mll_floor"] for bar in result["account_path"]] == [
        "97000.00",
        "97000.00",
        "97000.00",
        "99000.00",
        "99000.00",
        "100000.00",
        "100000.00",
        "100000.00",
    ]
    assert result["terminal"]["mll_floor"] == "100000.00"


def test_mll_touch_on_realized_or_unrealized_equity_blows_immediately() -> None:
    realized = {
        "schema_version": 1,
        "fixture_id": "realized-mll-touch",
        "bars": [
            _bar("2026-07-20T19:00:00+00:00", realized_pnl="-3000.00"),
        ],
    }
    unrealized = {
        "schema_version": 1,
        "fixture_id": "unrealized-mll-touch",
        "bars": [
            _bar(
                "2026-07-20T19:00:00+00:00",
                action="enter",
                contract="NQ",
                quantity=1,
                side="long",
                mark_ticks="-600",
                pending_orders=2,
            ),
        ],
    }

    realized_result = replay_account_fixture(_config(), realized, ROOT.parent)
    unrealized_result = replay_account_fixture(_config(), unrealized, ROOT.parent)

    assert realized_result["terminal"]["status"] == "BLOW"
    assert realized_result["terminal"]["balance"] == "97000.00"
    assert unrealized_result["terminal"]["status"] == "BLOW"
    assert unrealized_result["account_path"][-1]["event"] == "MLL_BLOW"
    assert unrealized_result["account_path"][-1]["position"] is None
    assert unrealized_result["account_path"][-1]["pending_orders"] == 0


def test_dll_touch_flattens_cancels_and_locks_until_5pm_without_blowing() -> None:
    fixture = {
        "schema_version": 1,
        "fixture_id": "dll-lockout-reset",
        "bars": [
            _bar(
                "2026-07-20T19:00:00+00:00",
                action="enter",
                contract="ES",
                quantity=1,
                side="long",
                mark_ticks="-158",
                pending_orders=3,
            ),
            _bar(
                "2026-07-20T19:03:00+00:00",
                action="enter",
                contract="ES",
                quantity=1,
                side="long",
            ),
            _bar(
                "2026-07-20T22:00:00+00:00",
                action="enter",
                contract="ES",
                quantity=1,
                side="long",
            ),
            _bar("2026-07-21T20:10:00+00:00"),
        ],
    }

    result = replay_account_fixture(_config(), fixture, ROOT.parent)
    path = result["account_path"]

    assert path[0]["event"] == "DLL_LOCKOUT"
    assert path[0]["position"] is None
    assert path[0]["pending_orders"] == 0
    assert path[1]["entry_accepted"] is False
    assert path[1]["entry_locked"] is True
    assert path[2]["entry_accepted"] is True
    assert path[2]["entry_locked"] is False
    assert path[3]["event"] == "SESSION_FLATTEN"
    assert result["terminal"]["status"] != "BLOW"


def test_profit_consistency_two_day_minimum_and_timeout_match_oracles() -> None:
    one_day = {
        "schema_version": 1,
        "fixture_id": "one-day-target",
        "bars": [_bar("2026-07-20T20:10:00+00:00", realized_pnl="6000.00")],
    }
    balanced_two_days = {
        "schema_version": 1,
        "fixture_id": "balanced-pass",
        "bars": [
            _bar("2026-07-20T20:10:00+00:00", realized_pnl="3000.00"),
            _bar("2026-07-21T20:10:00+00:00", realized_pnl="3000.00"),
        ],
    }
    inconsistent = {
        "schema_version": 1,
        "fixture_id": "inconsistent",
        "bars": [
            _bar("2026-07-20T20:10:00+00:00", realized_pnl="4000.00"),
            _bar("2026-07-21T20:10:00+00:00", realized_pnl="2000.00"),
        ],
    }
    timeout = {
        "schema_version": 1,
        "fixture_id": "twenty-day-timeout",
        "bars": [_bar(f"2026-07-{day:02d}T20:10:00+00:00") for day in range(1, 21)],
    }

    one_day_result = replay_account_fixture(_config(), one_day, ROOT.parent)
    pass_result = replay_account_fixture(_config(), balanced_two_days, ROOT.parent)
    inconsistent_result = replay_account_fixture(_config(), inconsistent, ROOT.parent)
    timeout_result = replay_account_fixture(_config(), timeout, ROOT.parent)

    assert one_day_result["terminal"]["status"] == "ACTIVE"
    assert one_day_result["terminal"]["trading_days"] == 1
    assert pass_result["terminal"]["status"] == "PASS"
    assert pass_result["terminal"]["consistency_ratio"] == "0.5"
    assert pass_result["account_path"][-1]["trading_days"] == 2
    assert pass_result["account_path"][-1]["session_days"] == 2
    assert pass_result["account_path"][-1]["best_day_profit"] == "3000.00"
    assert pass_result["account_path"][-1]["consistency_ratio"] == "0.5"
    assert inconsistent_result["terminal"]["status"] == "ACTIVE"
    assert inconsistent_result["terminal"]["consistency_ratio"].startswith("0.666")
    assert timeout_result["terminal"]["status"] == "TIMEOUT"
    assert timeout_result["terminal"]["session_days"] == 20


def test_mini_micro_equivalence_tick_economics_fees_and_one_time_costs() -> None:
    def fixture(contract: str, quantity: int) -> dict[str, object]:
        return {
            "schema_version": 1,
            "fixture_id": contract,
            "bars": [
                _bar(
                    "2026-07-20T19:00:00+00:00",
                    action="enter",
                    contract=contract,
                    quantity=quantity,
                    side="long",
                ),
                _bar("2026-07-20T20:10:00+00:00", action="exit", mark_ticks="4"),
            ],
        }

    mini = replay_account_fixture(_config(), fixture("ES", 1), ROOT.parent)
    micro = replay_account_fixture(_config(), fixture("MES", 10), ROOT.parent)

    assert mini["account_path"][0]["position"]["account_units"] == 10
    assert micro["account_path"][0]["position"]["account_units"] == 10
    assert mini["account_path"][0]["balance"] == "100000.00"
    assert micro["account_path"][0]["balance"] == "100000.00"
    # Four gross ticks less two adverse ticks per side and the pinned RT fee.
    assert mini["terminal"]["balance"] == "100021.22"
    assert micro["terminal"]["balance"] == "100012.80"


@pytest.mark.parametrize(
    ("contract", "quantity", "message"),
    (
        ("MZB", 10, "unsupported contract mapping"),
        ("ZB", 10, "configured mini profile"),
        ("MES", 1, "configured micro profile"),
    ),
)
def test_unsupported_mappings_and_profile_quantities_fail_closed(
    contract: str, quantity: int, message: str
) -> None:
    fixture = {
        "schema_version": 1,
        "fixture_id": "invalid-contract",
        "bars": [
            _bar(
                "2026-07-20T19:00:00+00:00",
                action="enter",
                contract=contract,
                quantity=quantity,
                side="long",
            )
        ],
    }

    with pytest.raises(RlAccountError, match=message):
        replay_account_fixture(_config(), fixture, ROOT.parent)


def test_replay_manifest_is_atomic_no_overwrite_and_byte_deterministic(tmp_path: Path) -> None:
    input_path = tmp_path / "fixture.json"
    input_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fixture_id": "deterministic",
                "bars": [_bar("2026-07-20T19:00:00+00:00", realized_pnl="1.25")],
            },
            sort_keys=True,
        )
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    write_account_replay_manifest(_config(), input_path, first, ROOT.parent)
    write_account_replay_manifest(_config(), input_path, second, ROOT.parent)

    assert first.read_bytes() == second.read_bytes()
    assert not list(tmp_path.glob(".*.json.*"))
    with pytest.raises(RlAccountError, match="already exists"):
        write_account_replay_manifest(_config(), input_path, first, ROOT.parent)


def test_cli_exposes_bounded_account_replay(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    expected = {"schema_version": 1, "terminal": {"status": "ACTIVE"}}
    config = _config()
    monkeypatch.setattr(cli, "load_rl_config", lambda _path: config)
    monkeypatch.setattr(cli, "write_account_replay_manifest", lambda *_args: expected)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mantis-v2",
            "rl-account-replay",
            "--config",
            "rl.toml",
            "--input",
            "fixture.json",
            "--output",
            "manifest.json",
        ],
    )

    cli.main()

    assert json.loads(capsys.readouterr().out) == expected
