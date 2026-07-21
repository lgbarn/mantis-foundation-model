# Mantis Foundation Model

This context covers reproducible foundation adaptation and downstream trading
research for the Mantis model family. MantisV2 is the only active model family.

## Language

**Foundation Adaptation**:
Supervised tuning of the pinned official foundation checkpoint on the validated
futures OHLCV corpus and NextLeg objectives.
_Avoid_: Pretraining, training from scratch

**Fresh Run**:
A new run identity initialized from the pinned official foundation checkpoint
with newly seeded task-specific parameters and no checkpoint from another run.
_Avoid_: Scratch run, reset run

**Run Identity**:
The immutable combination of experiment configuration, seed, source, dependency
lock, data, upstream model, initialization, and runtime contract that owns one
artifact directory.
_Avoid_: Job name, output folder

**Stage Identity**:
The immutable definition of one producer stage and the exact upstream artifact
identities it consumes.
_Avoid_: Step name, task ID

**Completed Artifact**:
A finalized, manifest-bound stage output that can no longer change under its
stage identity.
_Avoid_: Finished folder, latest output

**In-Progress Checkpoint Pointer**:
The atomic reference to the latest complete resumable checkpoint within an
active run whose identity is fixed but whose state is not yet finalized.
_Avoid_: Latest artifact, mutable run identity

**Artifact Chain**:
A sequence of manifest-bound outputs in which each stage verifies and records
the exact identity of every upstream input it consumes.
_Avoid_: Results folder, cache chain

**Transfer Bundle**:
A content-addressed collection of files whose relative paths, byte sizes, and
SHA-256 values are verified before it can become an input or backup.
_Avoid_: Upload, sync folder

**Platform Config**:
The versioned non-secret contract for remote resources, lifecycle limits,
storage, qualification, and spend policy.
_Avoid_: Cloud settings, infrastructure flags

**Experiment Config**:
The versioned scientific definition of model, data, objective, seed, and
promotion behavior independent of a specific machine.
_Avoid_: Training arguments, job config

**Local Config**:
The ignored machine-owned mapping for local paths and other non-secret operator
settings that do not define an experiment.
_Avoid_: Environment config, personal overrides

**Inventory Snapshot**:
A timestamped read-only record of relevant remote resources, availability,
prices, and billing state used to evaluate a launch intent.
_Avoid_: Cloud status, resource list

**Launch Intent**:
A requested remote workload and resource shape before policy, inventory, and
spend evaluation.
_Avoid_: Pod request, launch command

**Launch Authorization**:
A human approval bound to one exact launch decision, price observation, cost
ceiling, and expiry.
_Avoid_: General approval, available budget

**Launch Decision**:
The durable allow-or-reject result produced from a launch intent, inventory
snapshot, spend ledger, and launch authorization.
_Avoid_: Plan output, approval result

**Pod Receipt**:
The durable identity and observed terms of one created Pod, linked to the exact
launch decision that authorized it.
_Avoid_: Pod response, deployment record

**Termination Receipt**:
The durable evidence that a Pod stopped accruing compute charges, including its
identity, reason, timestamp, and reconciled cost.
_Avoid_: Stop result, deletion log

**Qualification**:
A bounded correctness, compatibility, resource, resume, and cost evaluation for
a platform path. Qualification is not model-quality evidence.
_Avoid_: Benchmark, smoke test

**Promotion Gate**:
A preregistered evidence threshold that a candidate must pass before it may
become the selected input to a later stage.
_Avoid_: Best result, winner check

**Sealed Holdout**:
The reserved 2026 partition that is inaccessible to ordinary training, tuning,
walk-forward selection, Optuna, Monte Carlo, and infrastructure qualification.
_Avoid_: Test set, validation set

**Spend Envelope**:
The maximum authorized cost for a named class of work, separate from account
balance and subject to its own launch and recovery rules.
_Avoid_: Budget balance, available credits

**Projected Spend**:
The conservative cost of a launch decision before resource creation.
_Avoid_: Estimated bill, expected price

**Reserved Spend**:
The portion of a spend envelope held by authorized or active work and therefore
unavailable to another launch.
_Avoid_: Pending charge, budget lock

**Actual Spend**:
The reconciled provider charge recorded after termination.
_Avoid_: Runtime estimate, projected cost

**Paid Gate**:
A human-owned decision that authorizes one exact, currently priced, bounded
resource mutation after every zero-cost prerequisite has passed.
_Avoid_: Apply step, launch permission
