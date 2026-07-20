from __future__ import annotations

import hashlib
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
        "ticker": "ES",
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
        "ticker": "ES",
        "bars": [
            _bar("2026-07-20T19:00:00+00:00", realized_pnl="-3000.00"),
        ],
    }
    unrealized = {
        "schema_version": 1,
        "fixture_id": "unrealized-mll-touch",
        "ticker": "NQ",
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
        "ticker": "ES",
        "bars": [
            _bar(
                "2026-07-20T19:00:00+00:00",
                action="enter",
                contract="ES",
                quantity=1,
                side="long",
                mark_ticks="0",
                realized_pnl="-1971.22",
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
    assert path[0]["marked_equity"] == "98000.00"
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
        "ticker": "ES",
        "bars": [_bar("2026-07-20T20:10:00+00:00", realized_pnl="6000.00")],
    }
    balanced_two_days = {
        "schema_version": 1,
        "fixture_id": "balanced-pass",
        "ticker": "ES",
        "bars": [
            _bar("2026-07-20T20:10:00+00:00", realized_pnl="3000.00"),
            _bar("2026-07-21T20:10:00+00:00", realized_pnl="3000.00"),
        ],
    }
    inconsistent = {
        "schema_version": 1,
        "fixture_id": "inconsistent",
        "ticker": "ES",
        "bars": [
            _bar("2026-07-20T20:10:00+00:00", realized_pnl="4000.00"),
            _bar("2026-07-21T20:10:00+00:00", realized_pnl="2000.00"),
        ],
    }
    timeout = {
        "schema_version": 1,
        "fixture_id": "twenty-day-timeout",
        "ticker": "ES",
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


@pytest.mark.parametrize(
    ("contract", "quantity", "ending_balance"),
    (
        ("ES", 1, "100021.22"),
        ("MES", 10, "100012.80"),
        ("NQ", 1, "100006.22"),
        ("MNQ", 10, "99997.80"),
        ("RTY", 1, "100006.22"),
        ("M2K", 10, "99997.80"),
        ("YM", 1, "100006.22"),
        ("MYM", 10, "99997.80"),
        ("GC", 1, "100015.68"),
        ("MGC", 10, "100000.80"),
        ("CL", 1, "100015.98"),
        ("MCL", 10, "100004.80"),
        ("ZB", 1, "100059.74"),
    ),
)
def test_each_contract_tick_economics_fee_and_one_time_cost_oracle(
    contract: str, quantity: int, ending_balance: str
) -> None:
    fixture = {
        "schema_version": 1,
        "fixture_id": contract,
        "ticker": {
            "MES": "ES",
            "MNQ": "NQ",
            "M2K": "RTY",
            "MYM": "YM",
            "MGC": "GC",
            "MCL": "CL",
        }.get(contract, contract),
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

    result = replay_account_fixture(_config(), fixture, ROOT.parent)

    assert result["account_path"][0]["position"]["account_units"] == 10
    assert result["account_path"][0]["balance"] == "100000.00"
    # Four gross ticks less two adverse ticks per side and the pinned RT fee.
    assert result["terminal"]["balance"] == ending_balance


@pytest.mark.parametrize(
    ("ticker", "contract", "quantity", "message"),
    (
        ("ZB", "MZB", 10, "unsupported contract mapping"),
        ("ZB", "ZB", 10, "configured mini profile"),
        ("ES", "MES", 1, "configured micro profile"),
    ),
)
def test_unsupported_mappings_and_profile_quantities_fail_closed(
    ticker: str, contract: str, quantity: int, message: str
) -> None:
    fixture = {
        "schema_version": 1,
        "fixture_id": "invalid-contract",
        "ticker": ticker,
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
                "ticker": "ES",
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


def test_input_identity_hashes_the_same_bytes_that_are_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = json.dumps(
        {
            "schema_version": 1,
            "fixture_id": "immutable-read",
            "ticker": "ES",
            "bars": [_bar("2026-07-20T19:00:00+00:00", realized_pnl="1.25")],
        },
        sort_keys=True,
    ).encode()
    replacement = json.dumps(
        {
            "schema_version": 1,
            "fixture_id": "replacement",
            "ticker": "ES",
            "bars": [_bar("2026-07-20T19:00:00+00:00", realized_pnl="9.99")],
        },
        sort_keys=True,
    ).encode()
    input_path = tmp_path / "fixture.json"
    input_path.write_bytes(original)
    real_loads = json.loads

    def replacing_loads(value):
        input_path.write_bytes(replacement)
        return real_loads(value)

    monkeypatch.setattr("mantis_v2.rl_account.json.loads", replacing_loads)

    result = write_account_replay_manifest(
        _config(), input_path, tmp_path / "manifest.json", ROOT.parent
    )

    assert result["fixture_id"] == "immutable-read"
    assert result["identities"]["input"] == hashlib.sha256(original).hexdigest()


def test_replay_rejects_switching_ticker_mid_combine() -> None:
    fixture = {
        "schema_version": 1,
        "fixture_id": "ticker-switch",
        "ticker": "ES",
        "bars": [
            _bar(
                "2026-07-20T19:00:00+00:00",
                action="enter",
                contract="ES",
                quantity=1,
                side="long",
            ),
            _bar("2026-07-20T19:03:00+00:00", action="exit", mark_ticks="4"),
            _bar(
                "2026-07-20T19:06:00+00:00",
                action="enter",
                contract="NQ",
                quantity=1,
                side="long",
            ),
        ],
    }

    with pytest.raises(RlAccountError, match="fixture ticker ES cannot trade NQ"):
        replay_account_fixture(_config(), fixture, ROOT.parent)


def test_first_bar_after_310pm_flattens_an_open_position() -> None:
    fixture = {
        "schema_version": 1,
        "fixture_id": "post-cutoff-flatten",
        "ticker": "ES",
        "bars": [
            _bar(
                "2026-07-20T20:09:00+00:00",
                action="enter",
                contract="ES",
                quantity=1,
                side="long",
            ),
            _bar("2026-07-20T20:11:00+00:00", mark_ticks="2", pending_orders=2),
        ],
    }

    result = replay_account_fixture(_config(), fixture, ROOT.parent)

    assert result["account_path"][-1]["event"] == "SESSION_FLATTEN"
    assert result["account_path"][-1]["position"] is None
    assert result["account_path"][-1]["pending_orders"] == 0


def test_session_flatten_entry_lock_clears_at_next_5pm() -> None:
    fixture = {
        "schema_version": 1,
        "fixture_id": "session-reset-after-flatten",
        "ticker": "ES",
        "bars": [
            _bar(
                "2026-07-20T20:09:00+00:00",
                action="enter",
                contract="ES",
                quantity=1,
                side="long",
            ),
            _bar("2026-07-20T20:10:00+00:00"),
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

    assert path[1]["event"] == "SESSION_FLATTEN"
    assert path[2]["entry_accepted"] is True
    assert path[2]["entry_locked"] is False


def test_numeric_zero_realized_pnl_does_not_create_a_trading_day() -> None:
    fixture = {
        "schema_version": 1,
        "fixture_id": "negative-zero",
        "ticker": "ES",
        "bars": [_bar("2026-07-20T20:10:00+00:00", realized_pnl="-0")],
    }

    result = replay_account_fixture(_config(), fixture, ROOT.parent)

    assert result["terminal"]["trading_days"] == 0


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
