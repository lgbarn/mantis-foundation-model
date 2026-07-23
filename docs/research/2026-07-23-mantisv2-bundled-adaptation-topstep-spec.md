# MantisV2 Bundled Adaptation and Topstep Pipeline Specification

Status: Accepted

## Problem Statement

The repository has a validated direct-full-fine-tune baseline, LoRA support,
foundation export, downstream embedding, PPO entry training, and Topstep
qualification, but it does not expose one small workflow that combines a
frozen-backbone head warm-start, parameter-efficient adaptation, and valid
Monte Carlo robustness. Running independent full experiments would repeat
expensive GPU work and delay evidence under the available budget.

The current confirmation bootstrap also samples weekly effects independently.
That preserves pairing but not adjacent market dependence, so it cannot be
described as a moving- or stationary-block uncertainty estimate.

## Solution

Build one cost-capped bundled candidate that starts from the pinned official
MantisV2 checkpoint, warms the existing NextLeg heads while the complete
encoder and zero-delta rank-8 LoRA adapters are frozen, then enables only those
LoRA adapters and the warmed heads under a fresh optimizer. The two phases
share one 10,000-update ceiling and one provenance lineage.

Reuse the completed A100 direct-full-fine-tune artifact as the comparison
baseline. A candidate that clears the existing representation and export gates
is embedded once, frozen, and consumed by the existing PPO entry-policy and
Topstep workflows. Monte Carlo is limited to contiguous train-fold episode
coverage, deterministic execution stress, and synchronized adjacent-week block
inference after replay. Exact chronological Topstep replay remains the
qualification authority.

Expose this path through one operator command that sequences existing commands
and stops at every existing gate. Do not introduce a general experiment engine,
new cloud control plane, or joint end-to-end gradient path.

## User Stories

1. As a researcher, I want one bundled training command, so that I can test the
   modern adaptation recipe without running three full GPU experiments.
2. As a budget owner, I want one fixed update ceiling, so that a phase change
   cannot silently multiply accelerator cost.
3. As a researcher, I want the task heads trained before representation
   adaptation, so that LoRA begins with a useful downstream mapping.
4. As an auditor, I want an explicit phase transition, so that optimizer state,
   trainable parameters, and artifact ancestry are reproducible.
5. As an operator, I want interrupted training to resume in the correct phase,
   so that paid work is not lost or repeated.
6. As a researcher, I want the existing A100 run reused as the baseline, so that
   comparison does not require another identical paid run.
7. As a downstream consumer, I want only a validated frozen export, so that PPO
   never consumes a stale or unqualified representation.
8. As a strategy owner, I want Trend Magic to remain direction authority, so
   that this experiment does not change the accepted trading decomposition.
9. As a risk researcher, I want Monte Carlo methods to preserve adjacent market
   dependence, so that confidence intervals do not treat independent weeks as
   interchangeable observations.
10. As a Topstep operator, I want exact chronological account replay to remain
    decisive, so that simulated uncertainty cannot manufacture pass evidence.
11. As an operator, I want one command to stop on a failed representation,
    export, embedding, PPO, or Topstep gate, so that downstream cost is not spent
    on an invalid parent artifact.
12. As a maintainer, I want minimal code that reuses existing deep modules, so
    that this experiment does not create a second training or orchestration
    framework.

## Implementation Decisions

- Add one bundled foundation adaptation mode with rank 8 and alpha 16 fixed.
  LoRA is installed at construction with zero initial delta.
- During the first phase, only the existing candle and nonlinear leg heads are
  trainable. This is named a frozen-backbone multi-head warm-start, not strict
  linear probing.
- At the phase boundary, retain model weights, enable rank-8 attention adapters
  and the heads, and create a fresh AdamW optimizer and fresh learning-rate
  schedule. Original encoder weights remain frozen.
- Store the active phase, phase-local progress, total update count, phase
  boundary identity, optimizer identity, and parent identities in resumable
  checkpoint provenance.
- Cap the bundle at 10,000 total updates. The warm-start phase receives at most
  2,000 updates; unused warm-start allowance transfers to LoRA. Early stopping
  may reduce cost but never expands the total budget.
- Use the existing NextLeg loss, four-timeframe corpus, batch semantics,
  validation selection, export parity, and artifact acceptance contracts.
- Reuse the validated direct-full-fine-tune evaluation as the reference after
  verifying its exact identities. Do not initialize the bundled candidate from
  its trained weights.
- Add one orchestration command that runs bundled training, validated export,
  frozen embedding, existing episode creation, PPO training, and Topstep
  confirmation in order. It delegates stage behavior to existing commands.
- Replace independent weekly resampling with synchronized adjacent-week moving
  blocks confined to each fold. Candidate and baselines use identical sampled
  blocks. Report the method and effective source-block count in the result.
- Trend Magic continues to own direction. The v1 PPO action remains entry
  acceptance only; deterministic exit and configured sizing remain frozen
  dependencies.
- MantisV2 and LoRA are frozen before embedding and receive no downstream RL
  gradients.

## Acceptance Criteria

1. One typed production configuration selects the bundled mode, rank 8,
   alpha 16, a 2,000-update warm-start ceiling, and a 10,000-update total
   ceiling while rejecting unknown or incompatible values.
2. Before the phase boundary, only the existing candle and leg heads change;
   the official encoder and all LoRA parameters remain byte-identical.
3. At the phase boundary, the complete model state is retained, only rank-8
   attention adapters plus heads become trainable, and a fresh optimizer and
   schedule start without loading warm-start optimizer state.
4. An uninterrupted run and an interrupted/resumed run produce identical phase
   transitions, update counts, selected metrics, and model tensors for a fixed
   deterministic fixture.
5. The combined run cannot exceed 10,000 optimizer updates, including across
   resume, and unused warm-start updates may transfer only to the LoRA phase.
6. Checkpoints and results identify the active phase, transition parent,
   per-phase updates, total updates, trainable parameter counts, source/config/
   data identities, and the existing provenance fields.
7. Existing LoRA adapter reload, merge, native-output parity, and export gates
   pass without weakening tolerances.
8. The downstream sequence consumes only the validated bundled export, embeds
   it once, preserves immutable manifest identities, and stops before each
   downstream stage when its parent gate fails.
9. The confirmation sampler emits synchronized adjacent-week blocks within
   folds, never crosses a fold, is deterministic for a fixed seed, and rejects
   an invalid block contract.
10. Monte Carlo result metadata accurately names the adjacent-week method and
    source/effective block counts; no result describes IID weekly sampling as a
    dependence-preserving block bootstrap.
11. Existing raw Topstep gates, chronological attempts, paired take-all and
    random-take baselines, mini/micro reporting, Trend Magic direction, exit,
    sizing, holdout, and exact rule behavior remain unchanged.
12. A zero-cost smoke executes the complete orchestration with fake/local
    dependencies, including a phase transition, export acceptance, downstream
    gate sequencing, and failure short-circuiting.
13. The focused tests and repository gate pass, and the operator documentation
    gives the exact single command, TensorBoard command, resume behavior,
    artifacts, stopping conditions, and cost boundary.

## Owned Surfaces

- Bundled foundation adaptation mode
- Bundled adaptation configuration
- Adaptation phase checkpoint identity
- Total update budget contract
- Bundled train-to-Topstep operator command
- Synchronized adjacent-week confirmation sampler
- Bundled pipeline operating guide

## Testing Decisions

- Build phase transitions test-first through the public training boundary.
- Use model-tensor hashes and optimizer construction records as independent
  oracles for freezing and fresh-optimizer behavior.
- Compare deterministic uninterrupted and resumed fixtures at both sides of the
  phase boundary.
- Exercise strict configuration with missing, unknown, negative, zero, and
  incompatible update budgets.
- Reuse existing LoRA targeting, merge parity, checkpoint provenance, export,
  embedding identity, episode, bounded RL resume, and Topstep tests.
- Test the adjacent-week sampler with hand-computable week sequences, fold
  boundaries, paired arms, repeated seeds, and invalid block lengths.
- Test orchestration with fake commands and call logs; paid infrastructure is
  not required for the repository gate.

## Blocked by

None - can start immediately.

## Out of Scope

- Independent rank-8, rank-16, strict-linear-probe, and full factorial runs
- Claiming which bundled component caused an observed improvement
- Updating original MantisV2 encoder weights during the LoRA phase
- Backpropagating PPO or Topstep rewards into MantisV2 or LoRA
- Learned direction, exit, or sizing in this experiment
- Synthetic Monte Carlo price histories or stitched trades
- Hyperparameter or architecture search in the sealed holdout
- Replacing the accepted RunPod v2 control path
- A general pipeline framework, workflow engine, or new paid service
- Live trading or profitability claims

## Research Basis

- **Source:** Plan-route issues #70-#74 and the supplied Kumar et al. LP-FT
  paper, 2026-07-23. **Verdict adopted:** test one bundled head-warm-start to
  rank-8-LoRA candidate against the existing direct-fine-tune artifact, then
  use bounded Monte Carlo around frozen PPO and exact Topstep replay. **Scope
  chosen:** one cost-capped diagnostic pipeline because the user explicitly
  prioritized speed and budget over component attribution.

| Decision | Backing finding | Confidence |
| --- | --- | --- |
| Warm both existing heads | The leg head is nonlinear, so strict LP-FT does not describe the local model | High |
| Recreate the optimizer | The trainable parameter topology changes at the phase boundary | High |
| Use rank-8 attention LoRA | Existing targeting covers all six fused attention projections with about 510K trainable parameters including heads | High |
| Reuse the A100 baseline | It is already validated and paid for; repeating it adds cost but no new control | High |
| Freeze before PPO | Stable embeddings preserve lineage and avoid repeated encoder work | High |
| Use adjacent-week blocks | Independent weekly resampling does not preserve serial market dependence | High |
| Keep exact replay authoritative | Monte Carlo uncertainty cannot create independent historical outcomes | High |

Research triggers become acceptance criteria 2-11. The inherited risks are
single-seed representation variance, uncertain optimal phase allocation, finite
historical regimes, simulator assumptions, and RunPod availability. The first
run remains a diagnostic candidate; formal promotion still requires the
predeclared confirmation evidence.

## Further Notes

The 20/80 update allocation is a cost policy, not a claim of optimality. The
single bundled test answers whether this complete recipe is worth further
confirmation. It cannot attribute a gain or failure to warm-starting, LoRA, or
Monte Carlo individually.
