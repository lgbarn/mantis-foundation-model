# MantisV2 RunPod Execution Specification

Status: Accepted

## Problem Statement

The MantisV2 foundation, Trend Magic downstream, and Topstep RL workflows are
implemented or planned around one local Mac and an external drive. Foundation
production is qualified only on Apple MPS, embedding rejects CUDA, RL is
qualified only on the current CPU host, TensorBoard is absent, and no
reproducible RunPod image, configuration, transfer, lifecycle, cost, or shutdown
implementation exists. Manually launching cloud resources would risk measuring
the wrong pipeline, leaking credentials, exceeding the funded spend envelope,
losing artifacts, weakening exact resume, or exposing the sealed holdout.

The operator needs a config-driven implementation that can prepare, qualify,
run, monitor, recover, and repatriate the complete MantisV2 workflow on RunPod
without routine source edits and without turning account balance into implicit
permission to spend.

## Solution

Build a local, fail-closed RunPod control plane around the existing MantisV2
pipeline. A pure launch-policy planner will evaluate a typed Platform Config,
Inventory Snapshot, Spend Ledger, and exact Launch Authorization into a durable
Launch Decision. A narrow Pod-control adapter may execute only an approved
decision and must return durable Pod and Termination Receipts. Terraform owns
stable non-secret resources, direct S3 transfer owns content-addressed bundles,
the immutable image owns the runtime, and existing MantisV2 commands own model
behavior.

Implementation proceeds through zero-cost dry runs before a human-approved
two-hour Secure Cloud H100 SXM qualification. Production remains locked until CUDA,
precision, resume, embedding, CPU-host, TensorBoard, artifact, and projected-cost
gates pass. Foundation and Trend Magic form one gated DAG; Topstep constrained
PPO, Optuna, Monte Carlo, and policy export form a separately CPU-qualified DAG.
The 2026 Sealed Holdout is never available to infrastructure or model selection.

## User Stories

1. As the operator, I want one typed Platform Config, so that changing paths,
   resource limits, deadlines, or candidate GPUs does not require source edits.
2. As a researcher, I want Experiment Configs separated from machine settings,
   so that scientific identity is stable across local and RunPod hosts.
3. As an operator, I want ignored Local Config for machine paths, so that local
   storage layout is explicit without entering versioned experiment identity.
4. As an operator, I want a read-only Inventory Snapshot, so that every launch
   decision uses current resources, prices, billing state, and datacenter facts.
5. As a budget owner, I want a deterministic Launch Decision, so that a Pod
   cannot be created from an unreviewed or stale request.
6. As a budget owner, I want Launch Authorization bound to one exact decision,
   ceiling, and expiry, so that general approval cannot be reused.
7. As an operator, I want only one live Mantis Pod, so that concurrent commands
   cannot multiply spend or corrupt a shared volume.
8. As an operator, I want a two-hour independent first-probe deadline, so that a
   hung trainer or lost SSH session cannot keep accruing charges.
9. As an auditor, I want Pod and Termination Receipts, so that every resource and
   charge can be reconciled with its authorization.
10. As a security owner, I want the account provisioning key confined to the
    local controller, so that workloads cannot mutate the account.
11. As a security owner, I want registry, S3, SSH, and Pod-scoped credentials
    treated as distinct concepts, so that no credential gains unintended scope.
12. As a maintainer, I want a digest-pinned Linux CUDA image, so that RunPod
    runtime behavior follows the committed source and dependency lock.
13. As a maintainer, I want image self-checks and recorded tool versions, so that
    driver, CUDA, Python, `uv`, and dependency drift fail before paid work.
14. As a data owner, I want Transfer Bundles with relative paths, sizes, and
    SHA-256 values, so that upload success or S3 ETags never imply eligibility.
15. As a data owner, I want mounted in-Pod verification and atomic promotion, so
    that partial or corrupt uploads never become training inputs.
16. As an operator, I want verified internal and external backups before remote
    deletion, so that RunPod and the unstable source drive are never sole copies.
17. As a researcher, I want a real-data CUDA qualification, so that synthetic
    smoke results are not mistaken for production compatibility.
18. As a researcher, I want CPU/CUDA FP32 parity, so that changing hardware does
    not silently change model semantics.
19. As a researcher, I want explicit BF16 qualification, so that faster training
    is used only when numerical and resume gates pass.
20. As an operator, I want batch-envelope and resource instrumentation, so that
    production retains GPU, RAM, disk, and cost headroom.
21. As an operator, I want TensorBoard through localhost-only SSH, so that I can
    observe long runs without exposing a public dashboard.
22. As an auditor, I want JSON metrics and manifests to remain authoritative, so
    that damaged or missing TensorBoard events do not destroy reproducibility.
23. As a researcher, I want a Fresh Run from the pinned official MantisV2 base,
    so that new adaptation does not inherit contaminated local checkpoints.
24. As a researcher, I want compute-matched 3-TF/4-TF comparison, so that adding
    5-minute context earns promotion rather than being assumed beneficial.
25. As a researcher, I want full-upstream, transformer-only, and LoRA candidates
    compared under registered gates, so that efficiency cannot displace accuracy.
26. As a researcher, I want multi-seed proper-score and negative-transfer gates,
    so that one favorable seed or instrument family cannot select the model.
27. As a release owner, I want validation-selected native/export parity, so that
    the published safetensors bundle exactly represents the accepted checkpoint.
28. As a strategy researcher, I want Trend Magic to own direction on every
    eligible state bar, so that the strategy is not limited to rare flip bars.
29. As a strategy researcher, I want fixed 3R labels kept separate from the
    execution trail after 2R, so that supervised outcomes remain stable.
30. As a strategy researcher, I want CUDA embedding parity and resumable shards,
    so that the promoted foundation can regenerate downstream features safely.
31. As a strategy researcher, I want chronological walk-forward proper-score
    gates, so that rejected heads cannot advance to simulation or holdout.
32. As a Topstep researcher, I want the 100K Combine rules and micro-contract
    economics preserved, so that simulation and RL optimize the actual task.
33. As a Topstep researcher, I want RL kept CPU-qualified and constrained, so
    that policy reward cannot hide account-blow cost or illegal actions.
34. As a Topstep researcher, I want validation-owned Optuna, development seeds,
    confirmation seeds, and synchronized-block Monte Carlo, so that promotion
    measures survival and pass probability without temporal leakage.
35. As a serving owner, I want exact policy export/reload behavior, so that live
    discrete actions match the promoted policy.
36. As a governance owner, I want the 2026 Sealed Holdout inaccessible until all
    definitions and gates are frozen, so that no tuning consumes it.
37. As an operator, I want exact checkpoint resume on a newly authorized Pod, so
    that termination does not require weakening provenance.
38. As an operator, I want durable failure evidence, so that a failed stage can
    be diagnosed without deleting partial output.
39. As a budget owner, I want projected, reserved, and actual spend reconciled,
    so that billing lag cannot authorize overlapping work.
40. As a maintainer, I want discoverable `just` workflows, so that ordinary
    operation changes configuration rather than Python or shell constants.
41. As an auditor, I want derived weights kept off Hugging Face while upstream
    licenses conflict, so that infrastructure does not imply redistribution.

## Implementation Decisions

- Use one repository context and the canonical terms in the root glossary.
- Use one on-demand Secure Cloud Pod at a time. The first qualification target
  is an H100 SXM with 80 GB VRAM, at least 8 vCPU, at least 32 GB RAM, and a
  two-hour hard deadline. Price and availability are observations, never
  configuration truths.
- Use one provider-sized 150 GB Standard network volume. Enforce a 120 GB
  high-water policy and at least 30,000,000,000 measured free bytes.
- Terraform owns the network volume and private Pod template. It does not create
  Pods until provider parity satisfies every required lifecycle field.
- A pinned Pod-control adapter owns launch, status, and terminate. The external
  interface is deliberately narrow: an approved Launch Decision enters; a Pod
  Receipt or Termination Receipt returns.
- The pure launch-policy planner is the primary seam for policy tests. It accepts
  Platform Config, Inventory Snapshot, Spend Ledger, and Launch Authorization
  and returns a durable allow-or-reject Launch Decision without side effects.
- Hold a local process lock on the single supported controller host across the
  fresh remote inventory check and create call. Reject other controller hosts.
- Keep Terraform, S3 transfer, SSH, training, and `just` outside Pod control.
  Reject generic multi-cloud abstraction, Terraform-created Pods, direct cloud
  calls from training code, shell-only policy, Serverless, Community Cloud, and
  Ansible.
- Use a digest-pinned Linux amd64 image with Python 3.12, pinned `uv`, frozen
  dependency sync, driver/runtime checks, and no data, weights, artifacts, or
  credentials.
- Keep the account provisioning key and S3 keys local. Use a registry-auth
  object for private pulls. Treat RunPod's automatically supplied Pod-scoped
  key as non-persistable workload metadata, never provisioning authority.
- Stage Transfer Bundles through the direct S3-compatible network-volume API.
  Verify path safety, source stability, sizes, and SHA-256; verify mounted bytes
  before atomic promotion. Rsync-over-SSH is a paid-time fallback only.
- Use terminate-and-recreate recovery for network-volume Pods. Each new Pod
  requires a new Launch Authorization and attaches the same verified volume.
- Preserve immutable Run and Stage Identities while active checkpoints, events,
  and their atomic pointer evolve. A Completed Artifact is immutable.
- Add typed precision identity and explicit FP32/BF16 behavior. BF16 promotion
  requires CPU/CUDA output, loss, gradient, update, export, and resume evidence.
- Persist TensorBoard events inside the active run, bind the server to
  `127.0.0.1:6006`, and access it only through SSH. JSON remains authoritative.
- Start Foundation Adaptation from the pinned official MantisV2 revision and
  weight digest. Never silently use random initialization.
- Compare full-upstream 3-TF and 4-TF arms under equal compute before screening
  full-upstream, transformer-only, LoRA-r8, and LoRA-r16 candidates. Screen
  seeds 42-44 and confirm seeds 42-46 using the accepted gates.
- Keep the 9 x 4 foundation universe and make 3-minute performance primary.
  A promoted 4-TF run preserves per-stream exposure unless qualification selects
  an equivalent effective-batch schedule.
- Keep Trend Magic direction, every eligible state bar, next-open entry, fixed
  3R label, and execution-only 2R/0.75R trail. Do not reuse rejected downstream
  identities or unreviewed 3-TF embeddings.
- Keep RL CPU-only until the actual host passes the accepted throughput gates.
  Existing issues #7-#12 and PR #19 own the RL tail; new infrastructure slices
  integrate rather than duplicate them.
- Keep inclusive final RL thresholds from the accepted parent contract:
  pass-rate lower bound at least 50% and blow-rate upper bound at most 1%.
  Reconcile stricter wording in issue #10 before its execution.
- Keep the sealed 2026 partition unavailable to hardware, precision, arm, seed,
  threshold, Optuna, walk-forward, and Monte Carlo selection.
- Enforce spend envelopes of $15 storage, $10 qualification, $100 production,
  and $25 protected recovery. Permit one live Pod, no automatic retry/fallback,
  no auto-pay, alerts at $50/$100, and ordinary launch rejection at $125.
- Do not publish upstream or derived weights while the license conflict remains.

## Acceptance Criteria

1. A strict Platform Config and ignored Local Config can describe the complete
   route without source edits and reject unknown, missing, or incompatible data.
2. A pure planner emits deterministic canonical Launch Decisions for allow and
   every rejection class without making network or filesystem mutations beyond
   its explicit output.
3. Launch Authorization is exact, expiring, single-use, and rejected when price,
   inventory, config, spend, or decision identity changes.
4. Concurrent local controllers cannot create two Pods, and remote inventory
   reconciliation rejects an existing managed or unmanaged Mantis Pod.
5. Pod creation accepts only an approved decision and emits a complete Pod
   Receipt; termination emits a complete reconciled Termination Receipt.
6. Independent deadline enforcement terminates a Pod after controller, trainer,
   SSH, or TensorBoard failure and never launches a retry.
7. Terraform plan contains only expected stable resources, has no secrets, and
   cannot duplicate Pod ownership.
8. The image builds reproducibly, passes runtime self-checks, records exact
   versions/digests, and contains no secret, data, or weight material.
9. Transfer rejects path traversal, symlinks, duplicates, changed sources,
   corrupt bytes, stale identities, and destructive synchronization.
10. Mounted SHA-256 verification and atomic promotion are required before a
    Transfer Bundle is consumable.
11. Remote deletion is refused until verified internal and external backups and
    explicit retention authorization exist.
12. TensorBoard events persist, parse, and remain localhost-only while JSON
    metrics/manifests independently reproduce run state.
13. CUDA qualification records environment, I/O, synchronized throughput,
    memory, OOM, checkpoint, data-wait, and cost metrics without holdout access.
14. CPU-FP32/CUDA-FP32 and qualified BF16 fixtures meet registered output, loss,
    gradient, update, export, and interrupted-resume tolerances.
15. CUDA embedding matches CPU fixtures, produces finite manifest-bound shards,
    and publishes a measured four-timeframe size/throughput projection.
16. The first paid qualification is impossible before zero-cost gates and exact
    human authorization, terminates within two hours, and stays within $10.
17. Production cannot launch until qualification passes and its conservative
    projection fits the remaining $100 production pool.
18. Foundation screening and confirmation use the pinned base, pre-holdout data,
    declared timeframes/modes/seeds, proper scores, and negative-transfer gates.
19. Foundation evaluation/export binds the validation-selected native checkpoint
    and passes safetensors parity before downstream use.
20. Downstream preparation preserves the accepted Trend Magic label/execution
    split, causality, purge/embargo, and explicit 4-TF producer identity.
21. Walk-forward failure writes durable evidence and prevents simulation or
    holdout; passing requires convergence plus log-loss and Brier baselines.
22. Target-host RL qualification passes action legality, deterministic replay,
    exact resume, 5,000 steps/second, and 100x mmap-latency headroom.
23. RL Optuna, confirmation, Monte Carlo, promotion, and export reuse issues
    #7-#12, preserve constrained cost behavior, and meet the inclusive accepted
    thresholds without ticker/profile aggregation hiding failure.
24. The Sealed Holdout cannot be opened by any ordinary remote workflow and can
    be consumed only once after a reviewed frozen release and exact unlock.
25. Every expensive stage is recoverable from authoritative manifests and two
    verified local copies without rewriting a Run or Stage Identity.
26. Operators can perform dry-run, plan, transfer, qualification, monitoring,
    TensorBoard access, termination, download, backup, and reconciliation through
    documented `just` workflows with exact identities and safe defaults.
27. No RunPod or Hugging Face publication path redistributes derived weights
    while the upstream license conflict remains unresolved.

## Owned Surfaces

- Platform Config and Local Config schemas
- Inventory Snapshot, Launch Intent, Launch Authorization, and Launch Decision
- Spend Ledger and spend reconciliation
- Pod-control adapter, Pod Receipt, and Termination Receipt
- Terraform stable-resource plan and ownership policy
- Reproducible CUDA image and runtime self-check
- Transfer Bundle, mounted verifier, and backup/retention authorization
- RunPod `just` workflow interface
- TensorBoard and resource-metric contract
- CUDA precision, parity, resume, and embedding Qualification bundles
- Portable foundation accuracy-matrix orchestration
- Portable Trend Magic downstream orchestration
- RunPod CPU qualification integration for existing Topstep RL surfaces
- RunPod operations, recovery, and artifact-repatriation documentation

## Testing Decisions

- Test external behavior at the highest seam: CLI exit status, canonical JSON
  plans/receipts/manifests, created files, and observable adapter calls.
- Table-test the pure launch-policy planner across price drift, expired/replayed
  authorization, stale inventory, budget reservation, billing lag, live-Pod
  conflicts, controller mismatch, deadline, and recovery-reserve cases.
- Test Pod control through an injected fake subprocess runner that speaks the
  pinned `runpodctl` JSON contract. Include malformed JSON, timeouts, partial
  creation, unknown Pod, idempotent reconciliation, and termination after
  controller failure.
- Race two local controllers against the same lock and remote fake inventory;
  the oracle is exactly one create call and one rejected Launch Decision.
- Scan every durable output and subprocess invocation for secret values.
- Test transfer through path and content fixtures, including traversal,
  symlinks, duplicates, source mutation, missing/mismatched retry, corrupt
  mounted bytes, atomic promotion, backup failure, and deletion refusal.
- Extend strict frozen-dataclass configuration and canonical-digest prior art.
- Extend existing provenance/no-overwrite, checkpoint/RNG resume, export parity,
  pipeline interruption, and holdout-access tests rather than creating parallel
  semantics.
- Run CUDA tests only on a qualified CUDA host; local unit tests must still
  exercise precision policy, metric schemas, and CPU reference fixtures.
- Treat money, lifecycle, transfer, precision, resume, and holdout slices as
  TDD work with independent oracles. Thin `just` wiring and documentation do not
  require a TDD label.
- Real paid Qualification is a human-owned acceptance issue, never an AFK unit
  test or automatic CI action.

## Blocked by

None - zero-cost implementation can start immediately. Paid qualification is
blocked by its implementation slices and a fresh Launch Authorization. RL
production remains blocked by PR #19 and the dependency chain in issues #7-#12.

## Out of Scope

- Creating or funding a RunPod resource during specification or AFK slices
- Random-weight MantisV2 base pretraining
- Mantis or Mantis+ implementation
- A generic multi-cloud orchestration framework
- Terraform-created Pods before verified provider parity
- Serverless, Community Cloud, Kubernetes, or Ansible
- Public TensorBoard or third-party hosted experiment tracking
- Changing Trend Magic direction, label, trail, Topstep 100K rules, micro
  economics, constrained-PPO objective, or accepted RL promotion thresholds
- Routine access to or tuning on the Sealed Holdout
- Publishing derived weights before license resolution

## Further Notes

The authoritative route decisions are GitHub issues #20-#28 and
`infra/runpod/IMPLEMENTATION_HANDOFF.md`. ADRs 0005-0007 record the topology,
ownership, and transfer tradeoffs. Existing RL issue bodies must be reconciled
with this accepted parent specification where wording differs; the parent spec
wins until both are deliberately updated.
