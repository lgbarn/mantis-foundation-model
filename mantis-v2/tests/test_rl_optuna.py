"""Contract tests for validation-owned Optuna search."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import optuna
import pytest
from mantis_v2 import cli
from mantis_v2.rl_config import load_rl_config
from mantis_v2.rl_optuna import (
    SEARCH_SPACE,
    PrunedTrial,
    SeedValidationOutcome,
    TrialEvaluation,
    TrialRequest,
    _atomic_json_idempotent,
    _canonical_sha256,
    _completed_evaluations,
    _completed_search_seed,
    _search_space_payload,
    _storage_url,
    _study_contract,
    derive_trial_identity,
    run_optuna_study,
    run_production_optuna_study,
    select_winner,
    validate_study_manifests,
)
from mantis_v2.rl_policy import PolicyVariant, ReturnNormalizers
from mantis_v2.rl_training import (
    PpoHyperparameters,
    _completed_timesteps,
    _generalized_advantages,
    _normalized_reward_advantages,
    _search_rollout_episodes,
    _Transition,
)

ROOT = Path(__file__).resolve().parents[1]


def test_search_space_is_immutable_and_contains_only_accepted_ppo_knobs() -> None:
    assert isinstance(SEARCH_SPACE, MappingProxyType)
    assert set(SEARCH_SPACE) == {
        "learning_rate",
        "rollout_length",
        "batch_size",
        "gae_lambda",
        "clip_range",
        "entropy_coefficient",
        "value_loss_coefficient",
        "max_grad_norm",
        "hidden_width",
    }
    forbidden = {
        "gamma",
        "reward",
        "exit",
        "sizing",
        "fees",
        "fold",
        "episodes",
        "holdout",
        "promotion_thresholds",
    }
    assert forbidden.isdisjoint(SEARCH_SPACE)
    assert SEARCH_SPACE["learning_rate"].low == 1e-4
    assert SEARCH_SPACE["learning_rate"].high == 1e-3
    assert SEARCH_SPACE["learning_rate"].log is True
    assert SEARCH_SPACE["rollout_length"].choices == (14, 28, 56)
    assert SEARCH_SPACE["rollout_length"].unit == "complete_episodes"
    assert SEARCH_SPACE["batch_size"].choices == (256, 512, 1024)
    assert (SEARCH_SPACE["gae_lambda"].low, SEARCH_SPACE["gae_lambda"].high) == (
        0.90,
        0.99,
    )
    assert SEARCH_SPACE["clip_range"].choices == (0.1, 0.2, 0.3)
    assert SEARCH_SPACE["entropy_coefficient"].low == 1e-4
    assert SEARCH_SPACE["entropy_coefficient"].high == 0.03
    assert SEARCH_SPACE["entropy_coefficient"].log is True
    assert SEARCH_SPACE["value_loss_coefficient"].choices == (0.25, 0.5, 1.0)
    assert SEARCH_SPACE["max_grad_norm"].choices == (0.3, 0.5, 1.0)
    assert SEARCH_SPACE["hidden_width"].choices == (64, 128, 256)
    assert list(SEARCH_SPACE) == [
        "learning_rate",
        "rollout_length",
        "batch_size",
        "gae_lambda",
        "clip_range",
        "entropy_coefficient",
        "value_loss_coefficient",
        "max_grad_norm",
        "hidden_width",
    ]
    assert _canonical_sha256(_search_space_payload()) == (
        "e660cf63ba6f159a4813c90d72b2eb1fd1fe20255ad668a0ff7f5e6d55cdcfb7"
    )
    with pytest.raises(TypeError):
        SEARCH_SPACE["gamma"] = object()  # type: ignore[index]


def test_trial_identity_derives_a_reproducible_three_seed_500k_schedule() -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-topstep-100k.toml")
    first = derive_trial_identity(config, "shared-ticker-value-v1", 0)
    repeated = derive_trial_identity(config, "shared-ticker-value-v1", 0)
    other = derive_trial_identity(config, "shared-ticker-value-v1", 1)

    assert first == repeated
    assert first != other
    assert first.trial_number == 0
    assert len(first.seeds) == 3
    assert len(set(first.seeds)) == 3
    assert first.timesteps_per_seed == 500_000
    assert first.total_timesteps == 1_500_000
    assert first.seeds == (136_966_313, 2_028_414_371, 1_707_253_496)
    assert first.proposal_seed == 764_857_380
    assert other.seeds == (1_364_155_168, 1_037_576_870, 855_980_109)
    assert derive_trial_identity(config, "shared-ticker-value-v1", 2).seeds == (
        1_511_916_982,
        1_425_734_077,
        180_491_388,
    )
    all_seeds = {
        seed
        for trial_number in range(30)
        for seed in derive_trial_identity(config, "shared-ticker-value-v1", trial_number).seeds
    }
    assert len(all_seeds) == 90


def _evaluation(
    trial_number: int,
    *,
    passes: int,
    blows: int = 0,
    median_days: float = 10.0,
) -> TrialEvaluation:
    outcomes = tuple(
        SeedValidationOutcome(
            partition="validation",
            seed=seed,
            attempts=10,
            passes=passes,
            blows=blows,
            median_pass_days=median_days,
        )
        for seed in (101, 102, 103)
    )
    return TrialEvaluation(trial_number=trial_number, outcomes=outcomes)


def test_winner_rejects_any_validation_blow_then_ranks_lcb_days_and_trial_number() -> None:
    winner = select_winner(
        (
            _evaluation(0, passes=7, median_days=10.0),
            _evaluation(1, passes=9, blows=1, median_days=1.0),
            _evaluation(2, passes=7, median_days=8.0),
            _evaluation(3, passes=7, median_days=8.0),
        )
    )

    assert winner.trial_number == 2
    assert winner.feasible is True
    assert winner.aggregate_blows == 0
    assert winner.pass_rate_lcb_95 > 0.0
    assert winner.median_pass_days == 8.0

    no_pass = tuple(
        SeedValidationOutcome("validation", seed, 10, 0, 0, None) for seed in (101, 102, 103)
    )
    assert (
        select_winner((TrialEvaluation(5, no_pass), TrialEvaluation(4, no_pass))).trial_number == 4
    )


def _schedule(path: Path, config_sha256: str, partition: str, end: str) -> None:
    start = "2025-01-01T00:00:00+00:00" if partition == "training" else "2025-07-01T00:00:00+00:00"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "rl-episode-schedule",
                "fold": 2,
                "partition": {
                    "name": partition,
                    "start": start,
                    "end": end,
                },
                "identities": {"config": {"sha256": config_sha256}},
            }
        )
    )


def test_study_manifests_expose_only_training_and_preholdout_validation(tmp_path: Path) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-topstep-100k.toml")
    training = tmp_path / "training.json"
    validation = tmp_path / "validation.json"
    _schedule(training, config.digest, "training", "2025-06-01T00:00:00+00:00")
    _schedule(validation, config.digest, "validation", "2025-12-31T23:59:59+00:00")

    manifests = validate_study_manifests(config, training, validation)

    assert manifests.fold == 2
    assert manifests.training_sha256 != manifests.validation_sha256
    _schedule(validation, config.digest, "test", "2025-12-31T23:59:59+00:00")
    with pytest.raises(Exception, match="validation partition"):
        validate_study_manifests(config, training, validation)
    _schedule(validation, config.digest, "validation", "2026-01-01T00:00:00+00:00")
    with pytest.raises(Exception, match="sealed holdout"):
        validate_study_manifests(config, training, validation)
    _schedule(validation, config.digest, "validation", "2025-12-31T23:59:59+00:00")
    forged = json.loads(validation.read_text())
    forged["partition"]["start"] = "2025-05-01T00:00:00+00:00"
    validation.write_text(json.dumps(forged))
    output = tmp_path / "must-not-exist"
    with pytest.raises(Exception, match="overlap"):
        run_optuna_study(
            config,
            training,
            validation,
            output,
            study_name="forged-v1",
            variant="shared_ticker_value",
            evaluator=lambda _request: pytest.fail("evaluator must remain unreachable"),
        )
    assert not output.exists()


def test_persistent_study_resumes_missing_trials_and_rejects_a_31st_atomically(
    tmp_path: Path,
) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-topstep-100k.toml")
    training = tmp_path / "training.json"
    validation = tmp_path / "validation.json"
    _schedule(training, config.digest, "training", "2025-06-01T00:00:00+00:00")
    _schedule(validation, config.digest, "validation", "2025-12-31T23:59:59+00:00")
    attempted: list[int] = []

    def evaluate(request: object) -> TrialEvaluation:
        identity = request.identity  # type: ignore[attr-defined]
        attempted.append(identity.trial_number)
        outcomes = tuple(
            SeedValidationOutcome("validation", seed, 10, 7, 0, 8.0) for seed in identity.seeds
        )
        return TrialEvaluation(identity.trial_number, outcomes)

    output = tmp_path / "study"
    progress = run_optuna_study(
        config,
        training,
        validation,
        output,
        study_name="shared-ticker-value-v1",
        variant="shared_ticker_value",
        evaluator=evaluate,
        maximum_trials_this_run=2,
    )

    assert progress["status"] == "incomplete"
    assert progress["attempted_trials"] == 2
    assert attempted == [0, 1]
    assert len(list((output / "ledger").glob("trial-*.json"))) == 2
    assert not (output / "winner.json").exists()

    completed = run_optuna_study(
        config,
        training,
        validation,
        output,
        study_name="shared-ticker-value-v1",
        variant="shared_ticker_value",
        evaluator=evaluate,
    )
    assert completed["status"] == "complete"
    assert completed["attempted_trials"] == 30
    assert attempted == list(range(30))
    assert (output / "winner.json").is_file()

    before = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    with pytest.raises(Exception, match="30-trial ceiling"):
        run_optuna_study(
            config,
            training,
            validation,
            output,
            study_name="shared-ticker-value-v1",
            variant="shared_ticker_value",
            evaluator=evaluate,
            maximum_trials_this_run=1,
        )
    after = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    assert after == before


def test_pruning_preserves_validation_only_partial_evidence(tmp_path: Path) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-topstep-100k.toml")
    training = tmp_path / "training.json"
    validation = tmp_path / "validation.json"
    _schedule(training, config.digest, "training", "2025-06-01T00:00:00+00:00")
    _schedule(validation, config.digest, "validation", "2025-12-31T23:59:59+00:00")

    def prune(request: object) -> TrialEvaluation:
        identity = request.identity  # type: ignore[attr-defined]
        evidence = tuple(
            SeedValidationOutcome("validation", seed, 10, 3, 0, 12.0) for seed in identity.seeds[:2]
        )
        raise PrunedTrial("validation LCB prune", evidence)

    output = tmp_path / "study"
    result = run_optuna_study(
        config,
        training,
        validation,
        output,
        study_name="shared-ticker-value-prune-v1",
        variant="shared_ticker_value",
        evaluator=prune,
        maximum_trials_this_run=1,
    )

    assert result["attempted_trials"] == 1
    ledger = json.loads((output / "ledger" / "trial-0000.json").read_text())
    assert ledger["state"] == "pruned"
    assert ledger["pruning_reason"] == "validation LCB prune"
    assert ledger["partial_validation"][0]["partition"] == "validation"


def test_split_resume_proposes_the_same_parameters_as_one_process(tmp_path: Path) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-topstep-100k.toml")
    training = tmp_path / "training.json"
    validation = tmp_path / "validation.json"
    _schedule(training, config.digest, "training", "2025-06-01T00:00:00+00:00")
    _schedule(validation, config.digest, "validation", "2025-12-31T23:59:59+00:00")

    def evaluate(request: object) -> TrialEvaluation:
        identity = request.identity  # type: ignore[attr-defined]
        return TrialEvaluation(
            identity.trial_number,
            tuple(
                SeedValidationOutcome("validation", seed, 10, 7, 0, 8.0) for seed in identity.seeds
            ),
        )

    split = tmp_path / "split"
    run_optuna_study(
        config,
        training,
        validation,
        split,
        study_name="deterministic-resume-v1",
        variant="shared_ticker_value",
        evaluator=evaluate,
        maximum_trials_this_run=11,
    )
    run_optuna_study(
        config,
        training,
        validation,
        split,
        study_name="deterministic-resume-v1",
        variant="shared_ticker_value",
        evaluator=evaluate,
    )
    continuous = tmp_path / "continuous"
    run_optuna_study(
        config,
        training,
        validation,
        continuous,
        study_name="deterministic-resume-v1",
        variant="shared_ticker_value",
        evaluator=evaluate,
    )

    split_params = [
        json.loads(path.read_text())["parameters"]
        for path in sorted((split / "ledger").glob("trial-*.json"))
    ]
    continuous_params = [
        json.loads(path.read_text())["parameters"]
        for path in sorted((continuous / "ledger").glob("trial-*.json"))
    ]
    assert split_params == continuous_params


def test_rollout_is_complete_balanced_supercycles() -> None:
    episodes = []
    for copy in range(8):
        for ticker in ("ES", "NQ", "RTY", "YM", "GC", "CL"):
            for profile in ("one_mini", "ten_micros"):
                episodes.append(SimpleNamespace(ticker=ticker, profile=profile, copy=copy))
        episodes.append(SimpleNamespace(ticker="ZB", profile="one_mini", copy=copy))

    rollout = _search_rollout_episodes(tuple(episodes), 56, seed=17, update=0)

    assert len(rollout) == 56
    assert len({id(episode) for episode in rollout}) == 56
    for start in range(0, 56, 14):
        supercycle = rollout[start : start + 14]
        assert Counter(episode.ticker for episode in supercycle) == {
            ticker: 2 for ticker in ("ES", "NQ", "RTY", "YM", "GC", "CL", "ZB")
        }
        for ticker in ("ES", "NQ", "RTY", "YM", "GC", "CL"):
            assert Counter(
                episode.profile for episode in supercycle if episode.ticker == ticker
            ) == {"one_mini": 1, "ten_micros": 1}
        assert [episode.profile for episode in supercycle if episode.ticker == "ZB"] == [
            "one_mini",
            "one_mini",
        ]


def test_gae_resets_at_episode_boundaries_and_timesteps_count_policy_decisions() -> None:
    def transition(reward: float, terminal: bool) -> _Transition:
        import numpy as np

        return _Transition(
            np.zeros(2, dtype=np.float32),
            0,
            0,
            0,
            0.0,
            0.0,
            0.0,
            np.ones(2, dtype=np.bool_),
            reward,
            0.0,
            terminal,
        )

    transitions = (
        transition(0.0, False),
        transition(1.0, True),
        transition(0.0, False),
        transition(4.0, True),
    )

    advantages = _generalized_advantages(
        transitions, signal="reward", old_value="old_value", gae_lambda=0.5
    )

    assert advantages.tolist() == pytest.approx([0.5, 1.0, 2.0, 4.0])
    assert _completed_timesteps(({"rollouts": {"policy_decisions": 123}},)) == 123


def test_reward_gae_uses_the_frozen_normalized_critic_basis() -> None:
    import numpy as np

    normalizers = ReturnNormalizers(("ES",))
    normalizers.update("ES", np.array([0.0, 2.0]))
    transitions = (
        _Transition(
            np.zeros(2, dtype=np.float32),
            0,
            0,
            0,
            0.0,
            0.2,
            0.0,
            np.ones(2, dtype=np.bool_),
            0.0,
            0.0,
            False,
        ),
        _Transition(
            np.zeros(2, dtype=np.float32),
            0,
            0,
            0,
            0.0,
            0.4,
            0.0,
            np.ones(2, dtype=np.bool_),
            1.0,
            0.0,
            True,
        ),
    )

    advantages = _normalized_reward_advantages(transitions, normalizers, 0.5)

    assert advantages.tolist() == pytest.approx([0.0, -0.4])


def test_search_parameters_bind_all_nine_training_knobs_and_reject_extras() -> None:
    parameters = {
        name: distribution.choices[0] if distribution.choices else distribution.low
        for name, distribution in SEARCH_SPACE.items()
    }
    parameters.update(
        learning_rate=1e-4,
        gae_lambda=0.90,
        entropy_coefficient=1e-4,
    )

    hyperparameters = PpoHyperparameters.from_search(parameters)

    assert hyperparameters.rollout_length == 14
    assert hyperparameters.batch_size == 256
    assert hyperparameters.hidden_width == 64
    with pytest.raises(Exception, match="search parameter keys"):
        PpoHyperparameters.from_search({**parameters, "gamma": 1.0})


def test_production_outcome_requires_checkpoint_bound_500k_evidence() -> None:
    outcome = SeedValidationOutcome("validation", 101, 10, 7, 0, 8.0)
    with pytest.raises(Exception, match="production evidence"):
        outcome.require_production_evidence()
    proven = SeedValidationOutcome(
        "validation",
        101,
        10,
        7,
        0,
        8.0,
        completed_timesteps=500_123,
        timestep_overshoot=123,
        checkpoint_sha256="a" * 64,
        training_manifest_sha256="b" * 64,
        validation_manifest_sha256="c" * 64,
        validation_artifact_sha256="d" * 64,
    )
    proven.require_production_evidence()


def test_metrics_only_search_seed_is_resumed_to_publish_its_manifest(tmp_path: Path) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-topstep-100k.toml")
    identity = derive_trial_identity(config, "metrics-crash-v1", 0)
    parameters = {
        name: distribution.choices[0] if distribution.choices else distribution.low
        for name, distribution in SEARCH_SPACE.items()
    }
    request = TrialRequest(
        identity,
        MappingProxyType(parameters),
        PolicyVariant.SHARED_TICKER_VALUE,
        tmp_path / "training.json",
        tmp_path / "validation.json",
        0,
        tmp_path / "trial",
        (),
        lambda _outcome: None,
    )
    seed_output = tmp_path / "trial" / "seed-0"
    seed_output.mkdir(parents=True)
    (seed_output / "metrics.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "identities": {
                    "seed": identity.seeds[0],
                    "search_trial_number": 0,
                    "search_seed_index": 0,
                    "ppo_hyperparameters": parameters,
                },
            }
        )
    )

    assert _completed_search_seed(seed_output, request, 0) is None


def test_validation_artifact_publication_is_idempotent_after_a_crash(tmp_path: Path) -> None:
    path = tmp_path / "validation" / "seed-0.json"
    payload = {"schema_version": 1, "stage": "rl-optuna-seed-validation", "seed": 101}

    _atomic_json_idempotent(path, payload)
    original = path.read_bytes()
    _atomic_json_idempotent(path, payload)

    assert path.read_bytes() == original
    with pytest.raises(Exception, match="immutable artifact mismatch"):
        _atomic_json_idempotent(path, {**payload, "seed": 202})


def test_production_wrapper_rejects_actual_episodes_reaching_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mantis_v2.rl_optuna as search

    config = load_rl_config(ROOT / "configs" / "rl-entry-topstep-100k.toml")
    training = SimpleNamespace(
        partition="training",
        fold=0,
        episodes=(
            SimpleNamespace(bars=(SimpleNamespace(timestamp=datetime(2025, 6, 1, tzinfo=UTC)),)),
        ),
    )
    validation = SimpleNamespace(
        partition="validation",
        fold=0,
        episodes=(
            SimpleNamespace(
                bars=(
                    SimpleNamespace(timestamp=datetime(2025, 12, 1, tzinfo=UTC)),
                    SimpleNamespace(timestamp=datetime(2026, 1, 2, tzinfo=UTC)),
                )
            ),
        ),
    )
    monkeypatch.setattr(search, "load_training_episodes", lambda *_args: training)
    monkeypatch.setattr(search, "load_episode_manifest", lambda *_args: validation)

    with pytest.raises(Exception, match="episodes reach the sealed holdout"):
        run_production_optuna_study(
            config,
            tmp_path / "training.json",
            tmp_path / "validation.json",
            tmp_path / "output",
            study_name="holdout-overlap-v1",
            variant="shared_ticker_value",
        )


def test_running_trial_resumes_same_plan_before_any_new_ask(tmp_path: Path) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-topstep-100k.toml")
    training = tmp_path / "training.json"
    validation = tmp_path / "validation.json"
    _schedule(training, config.digest, "training", "2025-06-01T00:00:00+00:00")
    _schedule(validation, config.digest, "validation", "2025-12-31T23:59:59+00:00")
    plans: list[dict[str, float | int]] = []

    def interrupt(request: TrialRequest) -> TrialEvaluation:
        plans.append(dict(request.parameters))
        raise KeyboardInterrupt

    output = tmp_path / "study"
    with pytest.raises(KeyboardInterrupt):
        run_optuna_study(
            config,
            training,
            validation,
            output,
            study_name="running-resume-v1",
            variant="shared_ticker_value",
            evaluator=interrupt,
            maximum_trials_this_run=1,
        )

    def complete(request: TrialRequest) -> TrialEvaluation:
        plans.append(dict(request.parameters))
        return TrialEvaluation(
            request.identity.trial_number,
            tuple(
                SeedValidationOutcome("validation", seed, 10, 7, 0, 8.0)
                for seed in request.identity.seeds
            ),
        )

    result = run_optuna_study(
        config,
        training,
        validation,
        output,
        study_name="running-resume-v1",
        variant="shared_ticker_value",
        evaluator=complete,
        maximum_trials_this_run=1,
    )

    assert result["attempted_trials"] == 1
    assert plans[0] == plans[1]
    assert len(list((output / "plans").glob("trial-*.json"))) == 1
    assert len(list((output / "ledger").glob("trial-*.json"))) == 1


def test_running_trial_resumes_durable_validation_intermediate(tmp_path: Path) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-topstep-100k.toml")
    training = tmp_path / "training.json"
    validation = tmp_path / "validation.json"
    _schedule(training, config.digest, "training", "2025-06-01T00:00:00+00:00")
    _schedule(validation, config.digest, "validation", "2025-12-31T23:59:59+00:00")

    def interrupt(request: TrialRequest) -> TrialEvaluation:
        request.report_validation(
            SeedValidationOutcome("validation", request.identity.seeds[0], 10, 7, 0, 8.0)
        )
        raise KeyboardInterrupt

    output = tmp_path / "intermediate-resume"
    with pytest.raises(KeyboardInterrupt):
        run_optuna_study(
            config,
            training,
            validation,
            output,
            study_name="intermediate-resume-v1",
            variant="shared_ticker_value",
            evaluator=interrupt,
            maximum_trials_this_run=1,
        )

    def complete(request: TrialRequest) -> TrialEvaluation:
        assert len(request.completed_validation) == 1
        outcomes = [*request.completed_validation]
        outcomes.extend(
            SeedValidationOutcome("validation", seed, 10, 7, 0, 8.0)
            for seed in request.identity.seeds[1:]
        )
        return TrialEvaluation(request.identity.trial_number, tuple(outcomes))

    result = run_optuna_study(
        config,
        training,
        validation,
        output,
        study_name="intermediate-resume-v1",
        variant="shared_ticker_value",
        evaluator=complete,
        maximum_trials_this_run=1,
    )
    assert result["attempted_trials"] == 1
    assert len(list((output / "intermediates" / "trial-0000").glob("seed-*.json"))) == 3


def test_terminal_ledger_resumes_tell_without_reexecuting_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mantis_v2.rl_optuna as search

    config = load_rl_config(ROOT / "configs" / "rl-entry-topstep-100k.toml")
    training = tmp_path / "training.json"
    validation = tmp_path / "validation.json"
    _schedule(training, config.digest, "training", "2025-06-01T00:00:00+00:00")
    _schedule(validation, config.digest, "validation", "2025-12-31T23:59:59+00:00")

    def evaluate(request: TrialRequest) -> TrialEvaluation:
        return TrialEvaluation(
            request.identity.trial_number,
            tuple(
                SeedValidationOutcome("validation", seed, 10, 7, 0, 8.0)
                for seed in request.identity.seeds
            ),
        )

    original_tell = search._tell_from_ledger
    monkeypatch.setattr(
        search,
        "_tell_from_ledger",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    output = tmp_path / "tell-resume"
    with pytest.raises(KeyboardInterrupt):
        run_optuna_study(
            config,
            training,
            validation,
            output,
            study_name="tell-resume-v1",
            variant="shared_ticker_value",
            evaluator=evaluate,
            maximum_trials_this_run=1,
        )
    monkeypatch.setattr(search, "_tell_from_ledger", original_tell)

    result = run_optuna_study(
        config,
        training,
        validation,
        output,
        study_name="tell-resume-v1",
        variant="shared_ticker_value",
        evaluator=lambda _request: pytest.fail("terminal ledger must suppress evaluator"),
        maximum_trials_this_run=1,
    )
    assert result["attempted_trials"] == 1


def test_full_state_cap_refuses_before_mutating_running_pruned_or_failed_trials(
    tmp_path: Path,
) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-topstep-100k.toml")
    training = tmp_path / "training.json"
    validation = tmp_path / "validation.json"
    _schedule(training, config.digest, "training", "2025-06-01T00:00:00+00:00")
    _schedule(validation, config.digest, "validation", "2025-12-31T23:59:59+00:00")
    manifests = validate_study_manifests(config, training, validation)
    output = tmp_path / "mixed-cap"
    output.mkdir()
    study_name = "mixed-cap-v1"
    study = optuna.create_study(
        study_name=study_name,
        storage=_storage_url(output / "study.sqlite3"),
        direction="maximize",
    )
    contract = _study_contract(
        config, manifests, study_name, PolicyVariant.SHARED_TICKER_VALUE, False
    )
    for key, value in contract.items():
        study.set_user_attr(key, value)
    for number in range(30):
        trial = study.ask()
        if number == 29:
            continue
        state = (
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
        )[number % 3]
        if state is optuna.trial.TrialState.COMPLETE:
            study.tell(trial, 0.5)
        else:
            study.tell(trial, state=state)

    before = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    with pytest.raises(Exception, match="30-trial ceiling"):
        run_optuna_study(
            config,
            training,
            validation,
            output,
            study_name=study_name,
            variant="shared_ticker_value",
            evaluator=lambda _request: pytest.fail("30th RUNNING trial must not resume"),
        )
    after = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    assert after == before


@pytest.mark.parametrize("tamper", ("parameters", "evaluation", "trial_number", "median_pass_days"))
def test_modified_ledger_cannot_participate_in_winner_selection(
    tmp_path: Path, tamper: str
) -> None:
    config = load_rl_config(ROOT / "configs" / "rl-entry-topstep-100k.toml")
    training = tmp_path / "training.json"
    validation = tmp_path / "validation.json"
    _schedule(training, config.digest, "training", "2025-06-01T00:00:00+00:00")
    _schedule(validation, config.digest, "validation", "2025-12-31T23:59:59+00:00")

    def evaluate(request: TrialRequest) -> TrialEvaluation:
        return TrialEvaluation(
            request.identity.trial_number,
            tuple(
                SeedValidationOutcome("validation", seed, 10, 7, 0, 8.0)
                for seed in request.identity.seeds
            ),
        )

    output = tmp_path / "tamper"
    run_optuna_study(
        config,
        training,
        validation,
        output,
        study_name="tamper-v1",
        variant="shared_ticker_value",
        evaluator=evaluate,
        maximum_trials_this_run=2,
    )
    ledger_path = output / "ledger" / "trial-0000.json"
    ledger = json.loads(ledger_path.read_text())
    if tamper == "parameters":
        ledger["parameters"]["batch_size"] = 999
    elif tamper == "evaluation":
        ledger["evaluation"]["outcomes"][0]["blows"] = 1
    elif tamper == "median_pass_days":
        for outcome in ledger["evaluation"]["outcomes"]:
            outcome["median_pass_days"] = 1.0
    else:
        ledger["evaluation"]["trial_number"] = 1
    ledger_path.write_text(json.dumps(ledger))
    study = optuna.load_study(
        study_name="tamper-v1", storage=_storage_url(output / "study.sqlite3")
    )
    contract = _study_contract(
        config,
        validate_study_manifests(config, training, validation),
        "tamper-v1",
        PolicyVariant.SHARED_TICKER_VALUE,
        False,
    )

    with pytest.raises(Exception, match="does not match"):
        _completed_evaluations(output, study, contract, config, "tamper-v1")


def test_cli_exposes_the_concrete_production_search_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sentinel = object()
    called: dict[str, object] = {}
    monkeypatch.setattr(cli, "load_rl_config", lambda _path: sentinel)

    def run(config: object, training: Path, validation: Path, output: Path, **kwargs: object):
        called.update(
            config=config,
            training=training,
            validation=validation,
            output=output,
            **kwargs,
        )
        return {"stage": "rl-optuna-search"}

    monkeypatch.setattr(cli, "run_production_optuna_study", run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mantis-v2",
            "rl-optuna-search",
            "--config",
            "rl.toml",
            "--training-manifest",
            "training.json",
            "--validation-manifest",
            "validation.json",
            "--output",
            "study",
            "--study-name",
            "shared-v1",
        ],
    )

    cli.main()

    assert json.loads(capsys.readouterr().out) == {"stage": "rl-optuna-search"}
    assert called == {
        "config": sentinel,
        "training": Path("training.json"),
        "validation": Path("validation.json"),
        "output": Path("study"),
        "study_name": "shared-v1",
        "variant": "shared_ticker_value",
    }
