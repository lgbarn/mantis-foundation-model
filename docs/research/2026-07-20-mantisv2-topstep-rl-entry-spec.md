# MantisV2 Topstep RL Entry Policy: Research-Driven Specification

## Status

Accepted for implementation on 2026-07-20. This authorizes repository code,
configuration, tests, and smoke artifacts required by the staged roadmap. It
does not authorize sealed-holdout access or live deployment before the declared
promotion gates pass.

## Provenance

- Generated: 2026-07-20
- Research scope: entry/exit/sizing decomposition, safe and constrained RL,
  multi-ticker transfer, Monte Carlo sampling, chronological evaluation,
  CPU/MPS tooling, current Topstep 100K rules, and the local FFM RL lineage.
- Local evidence: `mantis-foundation-model` at `6209969`; FFM RL evidence from
  `Futures-Foundation-Model` `origin/main` at `97a3992` because that code is not
  present in the divergent local checkout. Latest fetched public FFM upstream is
  `upstream/main` at `1982654`; it intentionally lacks the restored RL lineage.
- External evidence: 32 primary papers plus official Topstep, Gymnasium,
  Stable-Baselines3, SB3-Contrib, PyTorch, and Optuna documentation.
- Access date for external sources: 2026-07-20.
- Known evidence gap: no published study directly evaluates a binary
  MantisV2/Trend-Magic futures entry policy under Topstep's path-dependent rules.

## TL;DR

- **Build one independently qualified RL entry policy, not a monolithic trading
  policy.** Trend Magic owns direction; `EntryPolicyV1` chooses `skip` or
  `enter`; deterministic exit and configured fixed sizing remain frozen
  dependencies.
- **Train a shared ticker-conditioned actor on one uniformly sampled ticker per
  episode.** Use a shared critic trunk with a ticker-specific value output and
  return normalizer; defer ticker-specific actors until paired OOS evidence
  proves negative transfer.
- **Use Monte Carlo random starts only inside each training fold.** They improve
  optimization coverage but are not independent market evidence. Promotion
  remains based on purged chronological walk-forward tests and one sealed 2026
  holdout.
- **Separate learning objectives from hard rules.** PPO may optimize Combine
  completion, but Topstep MLL, legal fills, configured mini/micro exposure,
  session close, Trend Magic direction, and action validity are deterministic
  constraints.
- **Treat PPO as a challenger, not assumed improvement.** FFM's prior PPO passed
  only 3 of 852 holdout attempts and busted 813, far below random-take and
  take-all. That is a negative result for the older FFM environment and joint
  action design, not for the newer MantisV2 representation. Mantis must reuse
  FFM concepts, not its environment, reward, policy, folds, or weights.
- **Qualify CPU first.** Official SB3 guidance recommends CPU for MLP PPO.
  MPS remains useful for embedding generation but is not the production RL path
  unless a repo-local benchmark and determinism audit prove otherwise.

## 1. Problem and goal

The immediate goal is to train a policy that learns whether to accept each
causal, Trend-Magic-directed 3-minute entry opportunity in order to complete a
Topstep 100K Trading Combine. It is not a price forecaster and does not inherit
qualification from the rejected logistic probe.

The policy must learn the path-dependent consequences of entry selection:

- preserving the maximum-loss cushion;
- progressing toward the profit target;
- controlling best-day concentration;
- avoiding low-quality entries when account state is fragile;
- completing the Combine rather than maximizing isolated trade accuracy.

The policy remains subordinate to deterministic execution and risk rules.
Observed historical success is evidence, never a guarantee of live success.

### 1.1 Preservation boundary

RL extends the completed MantisV2 pipeline. It does not discard or retrain the
foundation model merely because the first linear probe failed. The following
assets and contracts remain authoritative inputs:

- the repaired July 2021-July 2026 Parquet corpus and its rollover validation;
- the jointly trained, validated, and exported MantisV2 foundation checkpoint;
- the audited 3,259,736-row, 3,840-wide embedding set;
- 1m/3m/15m context with 3m decisions;
- every-eligible-bar Trend Magic direction and candidate semantics;
- closed-bar to next-open causal execution;
- 24/3/3 purged walk-forward folds and the sealed 2026 holdout;
- the strict config, source, data, artifact, and export-parity chain;
- the deterministic Topstep account rules after upgrading them to bar-level RL
  transitions.

The rejected logistic result proves only that the tested linear probability head,
label, and configuration did not beat its proper-score baselines. It does not prove
that the newer MantisV2 embeddings contain no information useful to a sequential,
account-conditioned RL objective. Conversely, using a newer foundation version is
not proof that RL will work; the entry policy must demonstrate incremental OOS value
while every upstream artifact remains immutable.

## 2. Locked scope decisions

| Decision | Locked v1 behavior | Rationale |
| --- | --- | --- |
| Policy decomposition | Entry, exit, and sizing are separate versioned modules | They act on different clocks and have different action spaces |
| Delivery order | Train entry first; define exit/sizing interfaces now; learn them later | Prevent coupled credit assignment and preserve attribution |
| Direction | Trend Magic owns long/short | The RL entry experiment tests acceptance, not a second strategy |
| Entry action | Binary `skip` or `enter` | Minimal valid action space |
| Candidate clock | Every eligible closed 3-minute Trend Magic state | Avoid the prior sparse flip-only behavior |
| Fill timing | No earlier than next eligible bar open | Preserve causal execution |
| Exit v1 | Deterministic, config-pinned | A named immutable dependency of entry training |
| Size v1 | One mini or 10 eligible micros, fixed for the complete episode | The actor supports both profiles but cannot choose size; learned sizing remains deferred |
| Account objective | Complete a Topstep 100K Combine | This is the primary user outcome |
| Initial account | Exact fresh Combine state for every episode | Prevent domain randomization from replacing rule fidelity |
| Holdout | January-July 2026 stays sealed | No reward, architecture, seed, or threshold tuning on holdout |

## 3. Research conclusions

### 3.1 Sequential frozen modules beat joint learning for v1

Entry, exit, and sizing are naturally different control problems. Entry acts
only while flat at eligible closes; exit acts only while positioned; sizing acts
only at accepted entries. A monolithic Cartesian action creates many invalid
combinations and makes failure attribution opaque. Invalid-action masking helps
policy gradients but does not solve coupled credit assignment [1].

The options framework describes an entry as a temporally extended action with an
initiation set and termination rule [2]. V1 uses that structure without learning
the hierarchy: `enter` starts a deterministic position-management option. Learned
termination and sizing are later policies whose parent-policy hashes are explicit
dependencies. Option-Critic demonstrates learnable options [3], but offers no
evidence that joint option discovery is preferable for this trading problem.

### 3.2 The task is constrained RL plus a deterministic shield

The learning objective and mechanically enforceable constraints are different:

- objective: maximize probability of passing within the declared horizon;
- terminal outcomes: `PASS`, `BLOW`, or `TIMEOUT`;
- safety cost: MLL breach indicator and minimum cushion path;
- hard shield: actions, fills, session, position, size, discontinuity, and rule
  legality.

CPO formalizes reward optimization subject to separate costs [4]. It does not
guarantee safety under model error, gaps, or live execution. Reward-gaming work
shows that a large scalar blow penalty cannot make a proxy unhackable [5]. Known
Topstep rules should therefore be deterministic state transitions and action
masks, not learned preferences. Analytic safety layers have outperformed penalty
shaping on their own benchmark assumptions [6]; here the case for a shield is
stronger because the legal rules are explicitly known.

### 3.3 Monte Carlo is training coverage, not independent evidence

Random ticker and start-date selection creates varied training episodes from a
fixed historical panel. It does not generate new market histories. Overlapping
episodes reuse bars, regimes, embeddings, and trade opportunities, so counting
them as independent trials is pseudoreplication.

Every training episode must be a contiguous chronological path wholly inside a
fold's training interval. Model selection stays in validation. Test folds are
replayed only after the candidate is frozen. Uncertainty must preserve temporal
and cross-ticker dependence through synchronized calendar-block resampling [7].
Backtest variants, rewards, architectures, and seed-selection rules all contribute
to data-snooping debt [8, 9].

### 3.4 Shared actor, ticker-specific value baselines

The seven tickers share action semantics, account objective, Trend Magic context,
and MantisV2 representation. That supports a task-conditioned shared actor, but
not a ticker-blind policy. Give actor and critic a learned categorical ticker
embedding.

Reward scale, reward density, and task competence can make one task dominate
multi-task actor-critic learning. PopArt's multi-task result supports one shared
policy with task-specific value outputs and normalization [10]. V1 therefore uses:

```text
Mantis embedding ---- projection -----+
causal market state -------------------+--> shared actor --> skip / enter
ticker embedding ----------------------+         |
causal account state ------------------+         +--> Trend Magic direction
                                                   remains immutable

same observation --> shared critic trunk --> value_ES
                                         --> value_NQ
                                         --> value_RTY
                                         --> value_YM
                                         --> value_GC
                                         --> value_CL
                                         `-> value_ZB
```

Each ticker has separate critic-return normalization. Only the active ticker's
value head updates for an episode; the shared actor learns from all tickers.
Ticker-specific actors are permitted only after a preregistered ablation shows
stable paired OOS harm from sharing. Gradient conflict is diagnostic, not a gate:
PCGrad reports benefits [11], while ForkMerge shows gradient conflict does not
reliably predict held-out transfer [12].

### 3.5 PPO is a baseline algorithm, not a conclusion

PPO is reasonable because this is a binary discrete policy interacting with a
fully specified simulator. Its original evidence is generic robotics and Atari,
not finance [13]. PPO is on-policy and sample inefficient; implementation details,
reward scale, seeds, and library behavior materially change results [14, 15].

This setup is not classic offline RL from a logged behavior policy because prices
are modeled as exogenous and the simulator can calculate both legal actions on
historical paths. It is still limited to finite historical regimes and inherits
all simulator assumptions. Offline CQL/BCQ is a separate later experiment only if
an immutable transition dataset with behavior-policy coverage is created [16-19].

Official SB3 documentation recommends CPU and vectorized environments for MLP PPO
[20]. MaskablePPO supports discrete masks but requires its matching evaluation
callback and an environment-provided mask under subprocess vectorization [21].

## 4. FFM reference audit

FFM provides useful reference concepts:

- a pure account simulator;
- chronological multi-symbol evaluation and one account-wide position;
- symbol/account observations;
- take-all and random-take baselines;
- persistent Optuna studies;
- seed confirmation and export parity.

Its implementation and policy must not be copied. FFM used an older foundation,
different signals, different instruments, different folds, and a policy that jointly
controlled veto, size, hold, and exit. Its result therefore cannot be transferred as
a verdict on MantisV2. It remains valuable negative evidence about those design
choices. The final FFM PPO report shows:

| Policy | Holdout pass evidence |
| --- | ---: |
| Selected PPO | 3 / 852 attempts (0.35%) |
| Selected PPO MLL busts | 813 / 852 (95.4%) |
| Random-take | 11.5% pass rate |
| Take-all | 5.2% pass rate |

The FFM training environment randomly sampled single trades while carrying a
path-dependent account state, used synthetic training-day mechanics that differed
from evaluation, allowed outcome horizons to cross fold boundaries, had no
validation-owned selection layer, and combined veto, size, hold, and exit actions.
Its nominal shuffle control always produced zero reward rather than a shuffled
outcome. These are root design failures in that experiment, not reasons to reject
MantisV2, discard its artifacts, or reject all RL.

In particular, FFM does not implement the Monte Carlo Combine training described
by the supplied framework. It uniformly samples prebuilt single-trade episodes
with replacement, randomly stitches their outcomes into one evolving account,
and advances a synthetic session after a fixed number of trades. Mantis instead
uses complete contiguous historical Combine episodes. FFM also has no
micro-aware Topstep simulator: its quantity is an untyped 1-10 full-contract
count and its tick table contains only NQ, ES, RTY, YM, GC, and SI. Micro support,
CL, and ZB economics are Mantis-owned additions that require independent fixtures.

Mantis retains its stronger 24/3/3 folds, 1,000-bar embargo, event-span purge,
sealed holdout, artifact identities, and fail-closed gates. Reuse FFM concepts
behind Mantis-owned interfaces and tests; do not copy code, policy weights,
reward constants, action spaces, folds, or artifacts.

## 5. Current Topstep 100K rule contract

Pin the effective rule date and source snapshot in every artifact. As of the
2026-07-20 research date, official Topstep documentation specifies:

| Rule | Value |
| --- | --- |
| Starting balance | $100,000 |
| Initial MLL floor | $97,000 |
| MLL distance | $3,000 |
| Profit target | $6,000 |
| Consistency target | Best day at or below 50% of total profit |
| Minimum time | Pass in as few as 2 trading days |
| Position ceiling | 10 minis or 100 micros account-wide |
| Trading day | 5:00 PM CT through 3:10 PM CT |
| Overnight holding | Prohibited; flat by 3:10 PM CT |
| MLL update | Ratchets from end-of-day balance and never moves down |
| MLL enforcement | Continuous against realized and unrealized P&L |
| MLL lock | Locks permanently at $100,000 |

The supported micro mappings are ES/MES, NQ/MNQ, RTY/M2K, YM/MYM, GC/MGC,
and CL/MCL. Topstep's permitted-products list has no micro counterpart for ZB,
so ZB is mini-only. Contract class, quantity, tick size, tick value, commissions,
and the 10:1 Topstep position-equivalence rule are explicit simulator inputs and
artifact provenance. Mini and micro P&L must never be inferred from an untyped
quantity.

The pinned TopstepX round-turn fee snapshot is:

| Mini | RT fee | Micro | RT fee |
| --- | ---: | --- | ---: |
| ES | $3.78 | MES | $1.22 |
| NQ | $3.78 | MNQ | $1.22 |
| RTY | $3.78 | M2K | $1.22 |
| YM | $3.78 | MYM | $1.22 |
| GC | $4.32 | MGC | $1.92 |
| CL | $4.02 | MCL | $1.52 |
| ZB | $2.76 | none | n/a |

This table replaces FFM's stale flat $1.40-per-side assumption. Fees are
versioned external inputs and must be reverified before every production
qualification.

Automation is allowed, but Topstep does not troubleshoot it or excuse errant
orders. The environment and live shield must treat rule drift as a versioned
contract change that invalidates qualification.

## 6. Detailed v1 methodology

### 6.1 Episode definition

One episode represents one fresh Topstep 100K Combine attempt:

1. Select one ticker uniformly from ES, NQ, RTY, YM, GC, CL, and ZB.
2. Select a valid start from that ticker's current fold training interval.
3. Initialize the exact fresh account state.
4. Replay contiguous 3-minute bars without splicing or shuffling.
5. At every eligible closed bar while flat, Trend Magic provides direction and
   the actor selects `skip` or `enter`.
6. Accepted entries fill at the next eligible open with pinned fees and adverse
   slippage.
7. While positioned, deterministic exit logic advances on every bar. The actor
   cannot reverse, add, or exit.
8. Mark equity every bar, enforce MLL continuously, update the MLL only at the
   session boundary, and force flat by 3:10 PM CT.
9. End on `PASS`, `BLOW`, or configured `TIMEOUT`.

The 100K account also uses Topstep's optional $2,000 soft Daily Loss Limit. If
session net P&L reaches -$2,000, the environment flattens the open position,
cancels pending actions, and masks new entries until the next 5:00 PM CT session.
This is a daily lockout, not an account blow or failed Combine.

The episode must not cross a fold boundary, data discontinuity, missing session,
or corpus provenance change. The sampled tuple `(fold,ticker,start,end,seed)` is
recorded.

V1 deployment uses one ticker for an entire Combine. The ticker is selected from
the set of separately qualified instruments before the attempt starts and is
recorded in the run identity; it cannot change mid-attempt. The actor is still
trained jointly across all seven tickers, so this boundary does not discard
cross-market representation learning. Simultaneous multi-ticker opportunity
arbitration and a shared account trading several instruments are deferred until
the entry-only policy qualifies.

### 6.2 Clock and discounting

The environment advances one 3-minute bar per `step`. While a deterministic trade
is open, only the no-op action is legal. This avoids silently treating an entry
that lasts 20 bars and a one-bar skip as equal-duration decisions. An event-driven
alternative is forbidden unless it implements semi-MDP reward accumulation and
`gamma^duration` explicitly [2].

Use `gamma=1.0` for the first finite-horizon terminal-objective baseline. Report
time-to-pass separately. A lower discount can unintentionally discount delayed
blows as well as reward faster passes.

### 6.3 Observation contract

All values are causal as of the closed decision bar:

- frozen MantisV2 embedding and exact export identity;
- learned categorical ticker embedding;
- Trend Magic direction, line separation, time since direction change, and
  eligibility/discontinuity flags;
- normalized causal return, ATR, volume, and time/session features;
- flat/position state, although v1 actor authority exists only while flat;
- fixed episode sizing profile, contract class, quantity, dollar stop risk, tick
  economics, and booked fee schedule;
- balance, marked equity, MLL floor, normalized remaining cushion, normalized
  target distance, daily P&L, best-day profit, consistency ratio, trading days,
  accepted trades, and time remaining;
- action-validity mask.

Do not consume the rejected logistic probability or the framework's undefined
`p_chop_entry`. Statistical transforms fit on training only and ship inside the
policy bundle. Dollar P&L and tick economics remain exact in the simulator even
when policy observations use ATR, return, target, or MLL fractions.

### 6.4 Action and deterministic dependencies

```text
EntryPolicyV1 actions: 0=skip, 1=enter
TrendDirectionV1:      long or short, read-only
ExitPolicyV0:          deterministic 2R-activated, 0.75R-giveback trailing stop
SizerV0:               one mini or 10 eligible micros, fixed per episode
RiskShieldV1:          validates entry, fill, position, session, MLL buffer,
                       discontinuity, and contract legality
```

An invalid `enter` is masked, not converted into an unrecorded action. Training,
evaluation, and serving use the same mask implementation.

`ExitPolicyV0` preserves the intended Supertrend trade-management contract while
Trend Magic owns direction. One R is the initial 0.5 ATR risk. The initial stop
remains at -1R until the favorable extreme first reaches +2R. The trail then
ratchets behind the favorable high-water mark by 0.75R and can only tighten. It
has no fixed profit target. A bar is first tested against the stop established
from prior completed bars; its favorable extreme may tighten the stop only for
the next bar. This prevents same-bar path assumptions from creating a favorable
fill. The position is also closed at the 120-bar horizon or the mandatory session
close. Costs and adverse slippage follow the instrument-native contract below.

### 6.5 Reward and cost contract

The unshaped baseline is:

- `PASS`: terminal reward 1;
- `BLOW`: terminal reward 0 and constraint cost 1;
- `TIMEOUT`: terminal reward 0;
- non-terminal transition: reward 0.

This directly estimates a pass-probability objective but creates sparse credit.
The first permitted shaping challenger is a versioned potential:

```text
r_shaped = r_terminal + gamma * Phi(next_state) - Phi(state)
Phi(terminal) = 0
```

`Phi` may use only causal normalized target progress and MLL cushion. Potential
shaping preserves optimal policies under its MDP assumptions [22]. Do not add
independent P&L, speed, trade count, drawdown, entry-quality, and terminal terms
without a new reward identity and ablation. Delayed-reward methods such as RUDDER
are deferred [23].

MLL breaches are recorded as costs and terminal outcomes. The deterministic
shield prevents knowingly illegal entries but cannot guarantee survival through
market gaps or misspecified slippage. `Zero observed blows` is evidence, not a
claim of zero probability.

Friction is booked once, in dollars, by the account simulator. The primary
training and evaluation assumption is one adverse tick per side plus the pinned
TopstepX product-specific round-turn fee. It uses versioned tick-size,
tick-value, and fee tables for every supported mini and micro rather than a
shared R deduction. Promotion also requires a two-adverse-ticks-per-side stress
run with the same fee snapshot. CL, ZB, and every micro specification must be
independently fixture-tested because the FFM table does not include them. No
other environment layer may deduct friction again.

### 6.6 Training sampler

- Treat one complete fresh-account, contiguous 20-trading-day Combine attempt as
  one Monte Carlo episode; terminate earlier only on `PASS` or `BLOW`.
- Choose ticker uniformly before choosing a start.
- For instruments with a permitted micro counterpart, choose the fixed sizing
  profile uniformly between one mini and 10 micros. ZB always uses one mini.
- Sample a valid contiguous training start uniformly within ticker, optionally
  stratified by preregistered train-only calendar regimes.
- Keep the starting balance, MLL cushion, rules, and sizing profile exact rather
  than randomizing them. Randomness covers ticker, valid start, declared
  execution stress, and training seed.
- Balance completed rollout transitions per ticker or apply explicit weights.
- Replay original bars chronologically. Do not stitch trades, generate synthetic
  sessions, shuffle bars, concatenate instruments, or cross boundaries.
- Use separate vector environments with recorded seeds.
- Fit embedding projection, feature normalization, and each ticker's critic-return
  normalization on training only.

### 6.7 Optuna selection protocol

Monte Carlo, PPO, and Optuna have separate responsibilities:

1. The Monte Carlo sampler selects complete contiguous training episodes.
2. PPO learns one candidate policy from those episodes.
3. Optuna proposes only declared PPO optimizer/model hyperparameters and scores
   the trained candidate on validation-owned episode schedules.
4. The frozen winning configuration is retrained from scratch across development
   and confirmation seeds before any test or holdout claim.

Use a persistent, resumable Optuna study with TPE sampling, an immutable search
space, trial-specific seeds, atomic trial artifacts, and a hard ceiling of 30
trials. Each search trial receives 500,000 PPO steps over three seeds. Any
validation account blow makes the trial infeasible; feasible trials rank first by
validation pass-rate lower confidence bound and then by median days to pass.
Pruning may use validation evidence only and must preserve every partial result.

Permitted search knobs are learning rate, rollout length, batch size, GAE lambda,
PPO clip range, entropy coefficient, value-loss coefficient, gradient norm, and
the predeclared actor/critic hidden width. Gamma, reward definition, Trend Magic,
exit mechanics, sizing profiles, Topstep rules, fees, slippage, folds, episode
horizon, ticker sampling, holdout, and promotion thresholds are locked outside
Optuna. Architecture and reward ablations use separate named studies so trial
counts and multiple-testing debt remain visible.

## 7. Evaluation and experiment discipline

### 7.1 Existing temporal structure

Retain 24 months train, 3 months validation, 3 months test, 3-month stride, a
1,000-bar embargo, event-span purging, and the January-July 2026 sealed holdout.
An RL episode is valid only when its full state lookback, entries, exits, and
terminal accounting remain within its owning partition.

### 7.2 Preregistered baselines

Route every baseline through identical episodes and account mechanics:

1. reject-all;
2. take-all eligible Trend Magic states;
3. seeded random-take matched to the learned policy's entry count;
4. current rejected logistic head as historical evidence;
5. a fixed `HistGradientBoosting` contextual entry baseline trained only on the
   training partition with its decision threshold selected only on validation;
6. independent per-ticker PPO;
7. shared ticker-conditioned PPO with shared critic;
8. shared ticker-conditioned PPO with ticker-specific critic heads.

The last three form the multi-ticker transfer ablation. A pooled ticker-blind
policy may be a diagnostic but cannot be the default production choice.

The preregistered v1 candidate is the shared ticker-conditioned actor with
ticker-specific critic heads. It qualifies on validation only if every seed and
fold completes without non-finite values or action collapse, every seed records
zero blows, and the shared-minus-independent pass-rate point difference and
paired one-sided 95% lower confidence bound are nonnegative both pooled and for
every ticker/profile. A missing or non-estimable comparison fails closed.
Failure ends v1 architecture qualification; it does not permit selecting a
different ablation after results are visible. Final numeric Topstep promotion
thresholds remain test-owned under section 7.5 and are not weakened or inferred
from this validation qualification.

### 7.3 Seed and uncertainty protocol

- Development: at least 5 complete training seeds per candidate.
- Final pre-holdout claim: 10 complete seeds if compute permits.
- Never select the best seed for reporting or serving without a preregistered
  rule; report all seeds, median, interquartile mean, worst seed, and intervals.
- Seed 42 is the preregistered serving checkpoint. It is frozen before test
  evaluation and may export only if it and every required confirmation seed pass
  all gates. A failed seed 42 ends the candidate; do not substitute another seed
  after observing test or holdout results.
- Use fixed, non-overlapping chronological episode anchors for headline account
  outcomes when the test interval permits.
- Report overlapping-start results only as stress coverage.
- Estimate market uncertainty with synchronized calendar-block bootstrap across
  all tickers; estimate seed uncertainty separately.
- Count every tried architecture, reward, hyperparameter space, and selection
  rule in the experiment ledger.

RL point estimates from a few runs can reverse conclusions; interval estimates,
performance profiles, and robust aggregates are required [24].

### 7.4 Required metrics

Report per fold, ticker, regime block, and seed:

- pass, blow, and timeout counts/rates;
- pass-rate lower confidence bound and blow-rate upper confidence bound;
- trading days and calendar duration to pass;
- accepted trades, participation rate, skip rate, and action entropy;
- P&L net of costs, expectancy, drawdown, and minimum MLL cushion;
- best-day profit and consistency outcomes;
- result differences versus all identical-episode baselines;
- slippage, commission, missed-fill, one-bar latency, gap, and adverse same-bar
  sensitivity;
- per-ticker transfer gain versus its independent-policy baseline.

Every headline metric and promotion clause is also reported separately for the
one-mini and 10-micro profiles. Aggregate success cannot hide a failing contract
class. ZB contributes only to the mini profile.

### 7.5 Promotion rules already fixed by engineering invariants

- No partition leakage, horizon crossing, or holdout access.
- All required seeds and folds complete; missing evidence fails closed.
- No NaN/Inf observations, rewards, values, logits, or gradients.
- Exact source, config, corpus, embedding, foundation, rule, normalizer, policy,
  episode schedule, dependency lock, and output hashes.
- No ticker is hidden by aggregate performance.
- No simulation or holdout promotion when a configured gate fails.
- Native and exported actor outputs and actions match on fixed fixtures.

The Topstep promotion gate strengthens the FFM reference without weakening its
raw thresholds:

- at least 300 fixed, non-overlapping chronological Combine attempts;
- pooled raw pass rate at least 60%;
- every training seed's raw pass rate at least 50%;
- one-sided 95% Wilson pass-rate lower bound at least 50%;
- exactly zero observed account blows;
- one-sided 95% Wilson blow-rate upper bound at most 1%; and
- strict pooled pass-rate improvement over matched random-take and take-all
  baselines.

The paired synchronized calendar-block bootstrap must also place the one-sided
95% lower confidence bound for the shared actor's pass-rate difference above
zero versus both random-take and take-all. Against the independently trained
per-ticker actor ablation, the shared actor must have a non-negative pooled and
per-ticker point difference and a non-negative one-sided 95% lower confidence
bound for every supported sizing profile. If sharing fails this gate, the shared
v1 candidate does not promote; aggregate performance cannot conceal negative
transfer on one ticker or contract class.

All clauses must pass. Confidence intervals do not replace the raw FFM gates,
and zero observed blows is not described as zero true risk.

## 8. Integration with this repository

### 8.1 Proposed directory layout

```text
mantis-v2/
|-- configs/
|   |-- rl-entry-smoke.toml
|   `-- rl-entry-topstep-100k.toml
|-- src/mantis_v2/
|   |-- rl_config.py
|   |-- rl_environment.py
|   |-- rl_account.py
|   |-- rl_observation.py
|   |-- rl_policy.py
|   |-- rl_training.py
|   |-- rl_evaluation.py
|   `-- rl_export.py
|-- tests/
|   |-- test_rl_account.py
|   |-- test_rl_environment.py
|   |-- test_rl_observation.py
|   |-- test_rl_training.py
|   `-- test_rl_export.py
`-- docs/
    `-- rl-entry-runbook.md
```

Keep code inside `mantis-v2`; no second model-family consumer justifies moving it
to `shared/`.

### 8.2 Proposed root commands

```text
just rl-smoke <config>
just rl-validate-environment <config>
just rl-build-episodes <config>
just rl-train <config>
just rl-evaluate <config>
just rl-export <config>
just rl-validated-export <config>
```

No command is valid until implemented and tested. All commands use `uv`; the FFM
document's `./venv/bin/python` commands are not portable into this repo.

### 8.3 Build-versus-buy

| Need | Proposed tool | Custom responsibility |
| --- | --- | --- |
| Environment protocol | Gymnasium | Causal bar replay and Topstep semantics |
| Masked PPO baseline | SB3-Contrib MaskablePPO | Ticker-conditioned actor and multi-head critic |
| Core policy utilities | Stable-Baselines3 | Reproducible orchestration and manifests |
| Search | Optuna persistent storage | Fold/seed-aware fail-closed objective |
| Embeddings | Existing MantisV2 artifacts | Memory-mapped projection and provenance |
| Account rules | Mantis-owned pure state machine | Exact current Topstep rule snapshot |
| Export | Native PyTorch, optional manual ONNX | Full preprocessing/action parity |

The user explicitly approved installation of the listed free dependencies.
Installation still requires a post-install version check. Offline-RL libraries
are not v1 dependencies, and no paid or metered service is authorized.

### 8.4 Core interfaces

```python
class EntryPolicy(Protocol):
    def action_logits(self, observation: EntryObservation) -> NDArray[np.float32]: ...
    def act(self, observation: EntryObservation, mask: NDArray[np.bool_]) -> int: ...

class ExitPolicy(Protocol):
    def step(self, state: PositionState, bar: CausalBar) -> ExitDecision: ...

class PositionSizer(Protocol):
    def size(self, state: AccountState, proposal: EntryProposal) -> int: ...

class RiskShield(Protocol):
    def entry_mask(self, state: EnvironmentState) -> NDArray[np.bool_]: ...

class TopstepEntryEnv(gymnasium.Env[EntryObservation, int]):
    def reset(self, *, seed: int | None = None, options: dict | None = None): ...
    def step(self, action: int): ...
    def action_masks(self) -> NDArray[np.bool_]: ...
```

The actual code must use concrete typed data structures and strict config
parsing. These signatures define responsibility, not an implementation stub.

### 8.5 Configuration groups

The TOML schema must reject unknown keys and include:

```toml
[rl.run]
name = "unique-run-id"
seed = 42
device = "cpu"
artifact_root = "/machine/local/path"

[rl.policy]
role = "entry"
algorithm = "maskable_ppo"
actions = ["skip", "enter"]
ticker_conditioning = true
critic = "ticker_specific_heads"
embedding_projection_dim = 256

[rl.episode]
ticker_mode = "one_per_episode"
ticker_sampling = "uniform"
account_start = 100000.0
timeout_trading_days = 20
randomize_starting_cushion = false

[rl.execution]
adverse_slippage_ticks_per_side = 1.0
stress_adverse_slippage_ticks_per_side = 2.0
fee_schedule = "topstepx-2026-07-20"
cost_booking = "account_simulator_only"

[rl.sizing]
actor_controls_size = false
episode_profiles = ["one_mini", "ten_micros"]
profile_sampling = "uniform_when_supported"
mini_quantity = 1
micro_quantity = 10
mini_only_instruments = ["ZB"]

[rl.topstep]
daily_loss_limit_enabled = true
daily_loss_limit_dollars = 2000.0
daily_loss_limit_action = "flatten_cancel_lockout"
daily_loss_limit_terminal = false

[rl.reward]
kind = "terminal_pass_with_blow_cost"
gamma = 1.0
potential_shaping = false

[rl.training]
development_seeds = [42, 43, 44, 45, 46]
confirmation_seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
serving_seed = 42
vector_environments = 7
smoke_timesteps = 50000
search_timesteps_per_seed = 500000
search_seeds = 3
maximum_search_trials = 30
development_timesteps_per_seed = 2000000
confirmation_timesteps_per_seed = 5000000
maximum_timesteps_per_seed = 10000000

[rl.evaluation]
market_uncertainty = "synchronized_calendar_block_bootstrap"
sealed_holdout_start = "2026-01-01T00:00:00+00:00"
minimum_raw_pass_rate = 0.60
minimum_seed_raw_pass_rate = 0.50
minimum_pass_rate_lcb_95 = 0.50
maximum_observed_blows = 0
maximum_blow_rate_ucb_95 = 0.01
minimum_chronological_attempts = 300
```

The existing data, foundation, strategy, walk-forward, and Topstep groups remain
authoritative and are referenced rather than duplicated.

## 9. Test and acceptance specification

### 9.1 Deterministic environment qualification

1. `check_env` and MaskablePPO environment checks pass.
2. Same seed yields identical episode schedule and transitions on qualified CPU.
3. Prefix/truncation tests prove observations do not change when future bars are
   appended or modified.
4. Close-formed decisions never fill before the next eligible open.
5. No episode lookback, trade, exit, or terminal state crosses a partition.
6. Rollover and discontinuity fixtures reset all required state.
7. MLL ratchets only at end of day and is checked against every bar's adverse
   marked equity.
8. MLL locks at the configured starting balance and never moves down.
9. Consistency, two-day minimum, target, session, and contract fixtures match the
   pinned rule snapshot.
10. Gap-through-stop and stop-versus-current-bar-extreme ordering are
    conservative and deterministic.
11. Take-all, reject-all, and seeded random policies match independent native
    account fixtures exactly.
12. Invalid actions are impossible under the same mask in training and serving.

### 9.2 Learning-pipeline qualification

13. Scalers, projections, critic statistics, and samplers fit training only.
14. Uniform ticker selection is statistically verified over a fixed schedule.
15. Each critic head updates only on its owning ticker; actor updates use all.
16. Interrupted training resumes only with matching source/config/data/policy
    provenance.
17. Every seed and fold emits an atomic manifest and durable failure evidence.
18. No best-seed-only reporting; aggregate and worst-seed metrics are present.
19. Reward and mask ablations detect reject-all and take-all collapse.
20. The experiment ledger records every tried candidate identity.

### 9.3 Export and live-parity qualification

21. Policy bundle includes observation schema, normalizers, ticker mapping,
    action map, mask contract, and every dependency hash.
22. Native reload reproduces logits and deterministic actions exactly.
23. Optional exported actor matches native logits within declared tolerance and
    produces identical actions on boundary, ticker, account, and random fixtures.
24. Rule/config mismatch fails closed before inference.
25. The live risk shield has final authority and can only reduce exposure.

### 9.4 Performance targets

| Operation | Initial target |
| --- | ---: |
| Memory-mapped observation fetch | p95 below one 3-minute decision interval by at least 100x |
| Live actor plus mask | p99 below 100 ms on CPU |
| Environment step | At least 5,000 aggregate steps/s across vector workers |
| Resume | No replay of completed atomic seed/fold artifacts |
| Export parity | Identical discrete action on 100% of fixtures |

Targets must be measured on this Mac before production qualification; they are
engineering budgets, not research claims.

## 10. Staged roadmap

### Stage 0 - environment and baselines

- Implement strict config, pure account state, causal bar environment, masks,
  episode schedules, and deterministic baselines.
- Add every environment qualification test before PPO training.
- **Proceed only when:** all 12 deterministic tests pass and native baseline
  parity is exact.

### Stage 1 - CPU smoke

- Add proposed free/open-source dependencies after explicit approval.
- Train a tiny MaskablePPO run on deterministic fixtures and a bounded real fold.
- Verify save/load, resume, metrics, and finite gradients.
- **Proceed only when:** smoke is deterministic on CPU and beats reject-all on a
  learnable synthetic fixture.

### Stage 2 - architecture ablation

- Compare independent actors, shared actor/shared critic, and shared actor with
  ticker-specific critic heads on identical folds, schedules, costs, and budgets.
- Use validation only for selection.
- **Proceed only when:** the selected design clears predeclared per-ticker and
  aggregate validation gates across all development seeds.

### Stage 3 - frozen walk-forward confirmation

- Freeze architecture, reward, hyperparameters, seed handling, and thresholds.
- Run all chronological test folds over 5 seeds, then 10 if compute permits.
- Stress slippage, latency, gaps, and rule boundaries.
- **Proceed only when:** all numeric promotion gates are satisfied with intervals
  and no ticker-specific failure is hidden.

### Stage 4 - one-time sealed holdout

- Review the immutable candidate and evidence package.
- Unlock January-July 2026 once.
- A failure ends the candidate and cannot be tuned away.
- **Proceed only when:** holdout gates pass exactly as preregistered.

### Stage 5 - learned exit, then sizing

- Learned exit work depends on the frozen entry-policy hash.
- Learned sizing starts only after entry and exit are frozen and qualified.
- Any upstream policy change invalidates downstream qualification.

## 11. Locked scope summary

No scope decisions remain unresolved. The user confirmed this complete
specification on 2026-07-20 and authorized implementation.

The episode timeout is locked at 20 trading days. The deterministic v1 exit is
locked to a 2R activation and 0.75R high-water giveback trail with no fixed
target. Primary friction is one adverse tick per side plus the pinned
product-specific TopstepX round-turn fee; qualification also evaluates two
adverse ticks per side.

Compute is locked to 50,000 smoke steps; at most 30 search trials with 500,000
steps over 3 seeds; 2 million development steps over 5 seeds; and 5 million
confirmation steps over 10 seeds. A still-improving confirmation learning curve
may continue, but no seed may exceed 10 million steps.

The production account uses the optional $2,000 Topstep DLL as a soft session
lockout. Hitting it flattens and blocks new entries until the next session but
does not terminate the Combine as a blow.

V1 deploys exactly one preselected qualified ticker per Combine. This is a v1
qualification boundary, not a permanent rejection of multi-ticker account
execution.

The same entry actor trains and qualifies under fixed one-mini and 10-micro
episode profiles. The actor never chooses quantity. Both profiles must report
and pass gates separately; ZB is mini-only.

## 12. Caveats and triggers

- **Negative local prior.** FFM's PPO failure makes environment and baseline
  qualification mandatory before expensive training.
- **Weak finance-specific evidence.** Multi-task and safe-RL results come mainly
  from games, robotics, and portfolio/market-making tasks.
- **Finite history.** Random episode starts cannot manufacture new regimes.
- **Rule drift.** A Topstep rule change invalidates account qualification.
- **Market impact.** Fixed-size mini/micro simulation can model fees/slippage but
  not guarantee live fills. Revisit impact modeling before learned sizing.
- **No universal threshold.** Research does not establish an acceptable Topstep
  pass or blow rate; those are product-risk decisions.
- **Abort trigger.** Halt RL v1 if one adequately budgeted, learning-curve-confirmed
  study cannot beat preregistered deterministic/no-skill baselines across folds
  and seeds. Do not broaden action space to rescue it.
- **Specialization trigger.** Add ticker-specific actor adapters only when paired
  OOS evidence repeatedly shows material negative transfer.
- **Algorithm trigger.** Consider offline CQL/BCQ only after a valid immutable
  transition dataset and coverage audit exist.

## 13. Completion table

| Research item | Status | Section |
| --- | --- | --- |
| Entry/exit/sizing decomposition | Complete | 3.1, 6.4 |
| Safe reward and hard constraints | Complete | 3.2, 6.5 |
| Shared versus ticker-specific policies | Complete, direct domain evidence absent | 3.4, 7.2 |
| Monte Carlo and chronological evaluation | Complete | 3.3, 6.6, 7 |
| PPO/tooling and CPU/MPS | Complete | 3.5, 8.3 |
| FFM reference audit | Complete | 4 |
| Current Topstep 100K rules | Complete as of 2026-07-20 | 5 |
| Numeric product-risk gates | Complete | 7.5, 11 |

## 14. References

1. Huang, S. and Ontanon, S. (2020), "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms," arXiv:2006.14171. https://arxiv.org/abs/2006.14171
2. Sutton, R. S., Precup, D., and Singh, S. (1999), "Between MDPs and Semi-MDPs," *Artificial Intelligence* 112(1-2):181-211. https://doi.org/10.1016/S0004-3702(99)00052-1
3. Bacon, P.-L., Harb, J., and Precup, D. (2017), "The Option-Critic Architecture," *AAAI 31*. https://doi.org/10.1609/aaai.v31i1.10916
4. Achiam, J. et al. (2017), "Constrained Policy Optimization," *ICML*, PMLR 70:22-31. https://proceedings.mlr.press/v70/achiam17a.html
5. Skalse, J. et al. (2022), "Defining and Characterizing Reward Gaming," *NeurIPS 35*. https://proceedings.neurips.cc/paper_files/paper/2022/hash/3d719fee332caa23d5038b8a90e81796-Abstract-Conference.html
6. Dalal, G. et al. (2018), "Safe Exploration in Continuous Action Spaces," arXiv:1801.08757. https://arxiv.org/abs/1801.08757
7. Politis, D. N. and Romano, J. P. (1994), "The Stationary Bootstrap," *JASA* 89(428):1303-1313. https://doi.org/10.1080/01621459.1994.10476870
8. White, H. (2000), "A Reality Check for Data Snooping," *Econometrica* 68(5):1097-1126. https://doi.org/10.1111/1468-0262.00152
9. Bailey, D. H. et al. (2015), "The Probability of Backtest Overfitting," *Journal of Computational Finance*. https://ssrn.com/abstract=2326253
10. Hessel, M. et al. (2019), "Multi-task Deep Reinforcement Learning with PopArt," *AAAI*. https://arxiv.org/abs/1809.04474
11. Yu, T. et al. (2020), "Gradient Surgery for Multi-Task Learning," *NeurIPS 33*. https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html
12. Jiang, J. et al. (2023), "ForkMerge: Mitigating Negative Transfer in Auxiliary-Task Learning," *NeurIPS 36*. https://proceedings.neurips.cc/paper_files/paper/2023/file/60f9118a849e8e9a0c67e2a36ad80ebf-Paper-Conference.pdf
13. Schulman, J. et al. (2017), "Proximal Policy Optimization Algorithms," arXiv:1707.06347. https://arxiv.org/abs/1707.06347
14. Engstrom, L. et al. (2020), "Implementation Matters in Deep Policy Gradients," *ICLR*. https://openreview.net/forum?id=r1etN1rtPB
15. Henderson, P. et al. (2018), "Deep Reinforcement Learning That Matters," *AAAI 32*. https://doi.org/10.1609/aaai.v32i1.11694
16. Levine, S. et al. (2020), "Offline Reinforcement Learning: Tutorial, Review, and Perspectives," arXiv:2005.01643. https://arxiv.org/abs/2005.01643
17. Fujimoto, S. et al. (2019), "Off-Policy Deep Reinforcement Learning without Exploration," *ICML*, PMLR 97:2052-2062. https://proceedings.mlr.press/v97/fujimoto19a.html
18. Kumar, A. et al. (2020), "Conservative Q-Learning for Offline Reinforcement Learning," *NeurIPS 33*. https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html
19. Seno, T. and Imai, M. (2022), "d3rlpy," *JMLR* 23(315):1-20. https://www.jmlr.org/papers/v23/22-0017.html
20. Stable-Baselines3, "PPO," official documentation. https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
21. SB3-Contrib, "Maskable PPO," official documentation. https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
22. Ng, A. Y., Harada, D., and Russell, S. J. (1999), "Policy Invariance Under Reward Transformations," *ICML-99*:278-287. https://people.eecs.berkeley.edu/~russell/publications.html
23. Arjona-Medina, J. A. et al. (2019), "RUDDER: Return Decomposition for Delayed Rewards," *NeurIPS 32*. https://proceedings.neurips.cc/paper/2019/hash/16105fb9cc614fc29e1bda00dab60d41-Abstract.html
24. Agarwal, R. et al. (2021), "Deep Reinforcement Learning at the Edge of the Statistical Precipice," *NeurIPS 34*. https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html
25. Yu, T. et al. (2020), "Meta-World," *CoRL*, PMLR 100:1094-1100. https://proceedings.mlr.press/v100/yu20a.html
26. Teh, Y. W. et al. (2017), "Distral," *NeurIPS 30*. https://papers.nips.cc/paper_files/paper/2017/hash/0abdc563a06105aee3c6136871c9f4d1-Abstract.html
27. Marzban, S. et al. (2023), "WaveCorr," *Operations Research Letters* 51(6):680-686. https://doi.org/10.1016/j.orl.2023.10.011
28. Haider, A. et al. (2022), "Multi-Asset Market Making via Multi-Task Deep Reinforcement Learning," LNCS 13164:353-364. https://doi.org/10.1007/978-3-030-95470-3_27
29. Gymnasium, "Create a Custom Environment," official documentation. https://gymnasium.farama.org/main/introduction/create_custom_env/
30. Optuna, `create_study` and pruning API, official documentation. https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html
31. Topstep, "Trading Combine Parameters," official help center. https://help.topstep.com/en/articles/8284197-trading-combine-parameters
32. Topstep, "What is the Maximum Loss Limit?" official help center. https://help.topstep.com/en/articles/8284204-what-is-the-maximum-loss-limit
33. Topstep, "Consistency at Topstep," official help center. https://help.topstep.com/en/articles/8284208-consistency-at-topstep
34. Topstep, "When and What Products Can I Trade?" official help center. https://help.topstep.com/en/articles/8284206-when-and-what-products-can-i-trade

## Methodology

Three parallel research tracks covered policy decomposition/safe RL, multi-task
transfer/evaluation, and tooling/FFM evidence. Primary papers and official docs
were preferred; generic trading blogs and vendor performance claims were excluded.
The FFM implementation and report were inspected from immutable `origin/main`
objects without checking out or modifying that repository. Contradictory evidence
was resolved by treating PPO as a separately gated hypothesis: the FFM failure is
a strong negative prior against its design, not proof that every causal RL entry
policy must fail.
