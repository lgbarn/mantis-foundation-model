# Direct-LoRA 3TF PPO challenger run plan

Date: 2026-07-24

## Status and purpose

This run is a diagnostic challenger for the completed direct-LoRA, three-timeframe
MantisV2 foundation. It does not replace the failed linear downstream gate and it
cannot authorize simulation, promotion, or sealed-holdout access by itself.

The immutable inputs are:

- Foundation export: `mantisv2-foundation-training-first-3tf-direct-lora-s42-v1`
- Foundation weights SHA-256: `536ae864a1fa292b13d6dc98c61c45f3ab15646dbb354dc7f25d5ed4bf0926f0`
- Downstream embedding run: `mantisv2-trend-magic-direct-lora-3tf-cuda-v1`
- Embedding manifest SHA-256: `bdcc8819c2d68efff7bd48efc3ffdf4ba02bb73cfe01793d7a23208075b0625a`
- Embedding rows: `3,259,736`
- Embedding feature width: `3,840`
- Embedding storage: `25,280,949,347` bytes in 400 atomic shard pairs
- PPO run: `rl-entry-topstep-100k-direct-lora-3tf-v3`
- PPO config: `mantis-v2/configs/rl-entry-topstep-100k-direct-lora-3tf-v3.toml`
- Sealed holdout begins: `2026-01-01T00:00:00+00:00`

The eight-fold logistic head completed but failed both quality gates:

- Weighted log loss: `0.6953924403163079`; constant baseline: `0.6931471805599453`
- Weighted Brier score: `0.25110196380572225`; constant baseline: `0.25`

PPO is therefore being tested as a separate nonlinear entry policy. The failed
linear head remains a comparison baseline. The holdout remains sealed.

## Policy ownership

| Concern | Owner |
| --- | --- |
| Long or short direction | Trend Magic |
| Take or skip a qualified setup | Maskable PPO entry policy |
| Initial risk | Fixed 1R stop |
| Exit | Activate at 2R, then trail with 0.75R giveback |
| Maximum holding period | 120 bars |
| Position size | Fixed episode profile: 1 mini or 10 micros; ZB is mini-only |
| Account rules | Deterministic Topstep 100K state machine |

PPO does not choose direction, stop distance, exit behavior, or arbitrary size.
Its action space is only `skip` or `enter` after the deterministic setup and rule
masks have been applied.

## Execution route

Run the bounded qualification on a short-lived RunPod Pod attached to network
volume `wbxrj0n0ru` at `/workspace`. Prefer a CPU Pod when the volume's datacenter
supports one. RunPod CPU Pods do not currently support this volume's `US-MO-1`
datacenter, so the current run uses the least-cost available co-located GPU Pod
while PPO still executes on CPU. This avoids a 25.28 GB cross-region transfer.

The embeddings already exist on the volume and their signed manifest records
absolute `/workspace` paths. Do not copy those shards to the Mac, rewrite the
manifest, or spend Pod time re-auditing the accepted Parquet corpus. The PPO
implementation is qualified on CPU, not MPS or CUDA.

The Pod must use an immutable Git commit, a frozen dependency sync, a virtual
environment under `/tmp`, and a hard termination deadline. The network volume is
used only for immutable inputs and run artifacts.

## Exact bounded qualification commands

Run these commands from the committed repository checkout on the Pod:

```bash
export UV_PROJECT_ENVIRONMENT=/tmp/mantis-v2-rl-venv
uv sync --frozen --no-dev

uv run --no-sync mantis-v2 rl-dry-run \
  --config mantis-v2/configs/rl-entry-topstep-100k-direct-lora-3tf-v3.toml

uv run --no-sync mantis-v2 rl-build-episodes \
  --config mantis-v2/configs/rl-entry-topstep-100k-direct-lora-3tf-v3.toml \
  --fold 0 \
  --partition training \
  --episodes 21

uv run --no-sync mantis-v2 rl-build-episodes \
  --config mantis-v2/configs/rl-entry-topstep-100k-direct-lora-3tf-v3.toml \
  --fold 0 \
  --partition validation \
  --episodes 21

uv run --no-sync mantis-v2 rl-validate-environment \
  --config mantis-v2/configs/rl-entry-topstep-100k-direct-lora-3tf-v3.toml \
  --training-manifest /workspace/mantis/runs/rl-entry-topstep-100k-direct-lora-3tf-v3/episodes/fold-00-training-seed-42.json \
  --validation-manifest /workspace/mantis/runs/rl-entry-topstep-100k-direct-lora-3tf-v3/episodes/fold-00-validation-seed-42.json \
  --output /workspace/mantis/runs/rl-entry-topstep-100k-direct-lora-3tf-v3/environment-validation.json

uv run --no-sync mantis-v2 rl-train \
  --config mantis-v2/configs/rl-entry-topstep-100k-direct-lora-3tf-v3.toml \
  --training-manifest /workspace/mantis/runs/rl-entry-topstep-100k-direct-lora-3tf-v3/episodes/fold-00-training-seed-42.json \
  --output /workspace/mantis/runs/rl-entry-topstep-100k-direct-lora-3tf-v3/training/shared-ticker-value-bounded \
  --variant shared_ticker_value \
  --target-updates 1
```

The 21 training and 21 validation episodes give each of the seven instruments
three deterministic episode starts in each partition. This stage proves binding,
episode construction, environment semantics, checkpoint creation, and safe resume.
It is not a performance claim.

The first dry-run attempt at commit `5793997` failed before creating a run artifact
because the RL config referenced the producer's base config, which did not include
the reusable embedding binding. The corrected plan preserves an effective producer
snapshot and a separate bound consumer config. Their embedding contract digest is
`329a056ac005889b19696000884a87677803db9654464b2c69ddfeeeb43b02f3`, matching
the completed embedding manifest exactly.

The first environment-validation attempt then failed before writing validation
output because the bound config used a binding-only run name. Historical-policy
loading derives the rejected logistic manifest from the downstream run identity,
so the bound config now retains the actual CUDA producer run name. The producer
and embedding contract digests remain unchanged. The completed v1 schedules stay
preserved under their original config identity; the corrected run uses v2.

The v2 environment-validation replay exposed a matched-random baseline bug: the
baseline tried only 75 percent and 100 percent entry probabilities, so early
path-dependent trades could block or blow the account before matching a sparse
learned policy's accepted-trade count. The v3 source searches a bounded,
deterministic probability and seed grid centered on learned participation while
still requiring an exact accepted-trade match. All v2 evidence remains preserved.

## Advancement gates

Advance beyond the bounded update only when all of the following are true:

1. Every configured source, lock, rule, dataset, embedding, foundation, and weight
   hash matches the immutable input.
2. Episode schedules remain pre-holdout and training and validation schedules do
   not overlap.
3. Environment validation passes action masks, deterministic replay, fee booking,
   stop and trail behavior, Topstep state transitions, and reward accounting.
4. The bounded update writes a provenance-complete checkpoint and can resume
   without changing its run identity.

After qualification, the planned order is:

1. Optuna search on development seeds only.
2. Retrain the selected parameters across the development seed set.
3. Run chronological and synchronized moving-block Monte Carlo validation.
4. Run confirmation seeds once.
5. Consider simulation only if all configured pass-rate and blow-rate bounds pass.

Do not open the sealed holdout, promote a policy, or start live simulation during
this diagnostic stage. Those are separate governed decisions.

## Resume and failure behavior

- Never start a second command with the same run identity while one is active.
- Preserve atomic episode manifests, validation output, checkpoints, and logs on
  the network volume after any failure.
- Resume training only with `--resume` and the same immutable config and upstream
  hashes.
- Reject a missing or mismatched upstream artifact; do not repair provenance by
  editing a manifest.
- If the bounded stage fails, record the exact error and terminate the Pod after
  the persistent artifacts are confirmed.
- If the bounded stage passes, confirm the checkpoint and manifests on the network
  volume, then terminate the Pod before reviewing the next paid stage.
