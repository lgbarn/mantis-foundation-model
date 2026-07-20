# Setup, dependencies, and hardware

This guide explains what you need before running the repository and how CPU,
Apple MPS, and NVIDIA CUDA behavior differs. It separates implemented capability
from production-qualified behavior.

## Support summary

| Workflow | CPU | Apple MPS | NVIDIA CUDA |
| --- | --- | --- | --- |
| Foundation synthetic smoke | Supported | Supported through `auto` | Supported through `auto` |
| Upstream checkpoint verification | Required execution device | Not used | Not used |
| Foundation production fine-tuning | Code path exists but no qualified recipe | Qualified production path | Code path exists but no committed recipe |
| Foundation checkpoint RNG state | Saved | Saved when available | Saved for all visible CUDA devices |
| Downstream synthetic smoke | Required | Not used | Not supported |
| Downstream production embedding | CPU fallback exists | Qualified production path | Not supported by downstream config |
| Walk-forward logistic model and simulation | Supported | Acceleration is not required | Acceleration is not required |

Do not read "code path exists" as "validated production support." The committed
production configs and measured operating experience are Apple MPS-specific.

## System prerequisites

You need:

- A Git client.
- Python 3.12. Python 3.13 and later are outside the declared range.
- `uv` for environment and lock-file management.
- `just` for repository commands.
- Network access for the first dependency sync.
- Network access or a populated Hugging Face cache for the first official-weight
  verification.
- Enough local or external storage for source data and generated artifacts.
- Legal access to the market data you configure.

The repository does not install `uv`, `just`, Git, Python, GPU drivers, Xcode
command-line tools, or NVIDIA drivers. An AI agent must obtain permission before
installing packages or changing the host environment.

Check the tools without changing the environment:

```bash
python3 --version
uv --version
just --version
git --version
```

The Python result must be 3.12.x for the declared project contract. The exact
versions of `uv`, `just`, and Git are not pinned. Record them with any production
run.

## Python dependency policy

The root workspace declares development tooling. The MantisV2 package declares
runtime libraries. `uv.lock` records the complete exact resolution.

### Direct runtime dependencies

| Dependency | Declared range | Purpose |
| --- | --- | --- |
| `mantis-tsfm` | `==1.0.0` | Pinned upstream architecture package |
| `torch` | `>=2.4,<3` | Neural model, optimization, and devices |
| `huggingface-hub` | `>=0.23,<2` | Immutable checkpoint retrieval |
| `numpy` | `>=1.26,<3` | Numeric arrays and stored embeddings |
| `pandas` | `>=2.2,<3` | Tabular data preparation |
| `pyarrow` | `>=25,<26` | Parquet artifacts |
| `safetensors` | `>=0.4,<0.5` | Portable model export |
| `scikit-learn` | `>=1.9,<2` | Scaling, logistic regression, and metrics |

### Development dependencies

| Dependency | Declared range | Purpose |
| --- | --- | --- |
| `ruff` | `>=0.12,<0.13` | Formatting and linting |
| `mypy` | `>=1.17,<2` | Static type checking |
| `pytest` | `>=8.4,<9` | Test execution |
| `pytest-cov` | `>=6.2,<7` | Test coverage |

Use `uv.lock` as the exact dependency source of truth. Do not hand-install a
different PyTorch build into the managed environment and still call the run
reproducible.

## Create the environment

From the repository root:

```bash
just sync
```

This runs `uv sync --all-packages`. It may download and install packages. Run it
only with installation authority and not while a protected production run relies
on an unchanged environment.

Confirm the managed interpreter and selected PyTorch capabilities:

```bash
uv run python -c "import platform, torch; print(platform.platform()); print(torch.__version__); print('mps', torch.backends.mps.is_available()); print('cuda', torch.cuda.is_available())"
```

This command imports PyTorch but does not train a model. On a resource-sensitive
active run, prefer inspecting its existing provenance rather than starting
another Torch process.

## Prepare data

The repository does not download or redistribute market data. The operator must
create the configured streams and comply with the data provider's license.

### Foundation data contract

The committed production config expects this logical layout:

```text
<corpus-root>/
|-- manifest.json
`-- market/
    |-- ES_1min.parquet
    |-- ES_3min.parquet
    |-- ES_15min.parquet
    |-- NQ_1min.parquet
    `-- ...27 symbol/timeframe combinations total
```

Required columns:

```text
datetime,open,high,low,close,volume
```

Requirements:

- `datetime` parses as a UTC timestamp.
- Timestamps are strictly sorted and unique within each stream.
- OHLCV columns contain finite numeric values.
- Each file follows the exact configured symbol and interval name.
- Every file matches its identity in the corpus manifest.
- Continuous-contract construction, ratio back-adjustment, sessions, repaired
  rollovers, and missing-bar policy are recorded in that manifest.

The local production corpus uses 9 symbols and 3 timeframes. Another user can use
a different corpus only through a new reviewed config and run identity. Minimum
anchor counts, split safety, and downstream assumptions must still pass.

## Use portable paths and run names

The committed production TOML files record this machine's paths. They are not
portable templates. For a new host or experiment:

1. Copy the nearest config to a new descriptive filename.
2. Set `run.name` to a unique experiment identity.
3. Set `run.artifact_root` to a writable generated-artifact location.
4. Set `data.root` to the prepared data location.
5. Update downstream `foundation.manifest_path` only after a validated export
   exists.
6. Retain `allow_overwrite=false` for non-disposable work.
7. Treat the copied config as a versioned experiment definition.

Example file copy:

```bash
cp mantis-v2/configs/nextleg-parquet-v2.toml mantis-v2/configs/nextleg-myhost-mps.toml
```

Edit the copy, never the config of an active or resumable run. Foundation
commands do not offer path overrides. Downstream commands support scalar
`--set` overrides, but committed configs remain preferable for reproducibility.

## Storage planning

Separate these storage classes:

| Storage class | Examples | Git policy |
| --- | --- | --- |
| Source | Python, TOML, tests, Markdown | Commit |
| Small fixtures | Deterministic synthetic test data | Commit when appropriate |
| Raw data | Futures DBN archives and repaired Parquet streams | Never commit |
| Transformed data | Candidate Parquet, embedding shards | Never commit |
| Native state | `.pt`, optimizer state, RNG state | Never commit; private |
| Deployment export | `.safetensors`, manifest | Store outside Git |
| Results | Metrics, predictions, simulation outputs | Store outside Git |
| Caches and secrets | Hub cache, environment files, tokens | Never commit |

The repaired corpus contains 40,251,760 rows across 27 Parquet streams. Row
count and compressed bytes are not measurements of resident training memory.
Measure the selected corpus and use a separate artifact volume with enough
headroom for transactional checkpoints, best/latest copies, embeddings, and
stage outputs.

## Understand device selection

Foundation config accepts `auto`, `cpu`, `mps`, and `cuda`.

```text
auto -> CUDA when available -> MPS when available -> CPU
```

An explicit unavailable `mps` or `cuda` fails. `require_accelerator=true` rejects
CPU. The production config explicitly requires MPS, so it never silently falls
back.

The downstream config accepts only `auto`, `cpu`, and `mps`.

```text
downstream auto -> MPS when available -> CPU
```

Downstream `auto` never selects CUDA. On a Linux/NVIDIA host it will use CPU
unless the implementation changes.

## Run on CPU

CPU is the portable correctness path for:

- Foundation synthetic smoke on hosts without an accelerator.
- Official checkpoint verification.
- Downstream synthetic smoke.
- Walk-forward logistic regression.
- Simulation.

Full foundation fine-tuning can be configured for CPU only when
`require_accelerator=false`, but no production CPU batch size or run-time estimate
is qualified. It is expected to be much slower than MPS or CUDA.

Changing the device or `require_accelerator` changes the config digest. A CPU
config cannot resume the committed MPS production checkpoint.

## Run on Apple MPS

Apple Metal Performance Shaders (MPS) is the qualified local acceleration path.

Committed production choices:

- Explicit `device="mps"`
- `require_accelerator=true` for foundation training.
- Foundation batch size 128.
- Zero DataLoader workers.
- Downstream embedding batch size 128.

The committed probe exercises 32 real-data optimizer updates and one validation
batch across all configured streams:

```bash
just probe-mps
```

On another host, copy both the production and probe configs, set portable paths
and a new probe run name, then invoke the copied probe explicitly:

```bash
uv run mantis-v2 probe --config mantis-v2/configs/nextleg-myhost-mps-probe.toml
```

The probe contract requires exactly 1 epoch, 32 training updates, 1 validation
batch, no resume, and synthetic data disabled. The probe writes disposable
artifacts. It is a qualification step, not a performance benchmark or substitute
for full training.

MPS differs from CUDA in supported kernels, memory behavior, and floating-point
results. The code requests deterministic algorithms with warnings enabled, so an
unsupported deterministic kernel warns instead of failing. Exact cross-device
numeric identity is not guaranteed.

## Run on CUDA

The foundation runtime has an explicit CUDA code path:

- Explicit `device="cuda"` can select an available NVIDIA device.
- DataLoader pinning and non-blocking transfers activate for explicit CUDA.
- RNG state for all visible CUDA devices is saved and restored.

Important limitations:

- No CUDA config is committed.
- No CUDA-specific probe is implemented.
- No batch-size or memory recipe is measured.
- No multi-GPU or distributed training exists.
- No mixed precision, autocast, or gradient scaler exists.
- Downstream embedding does not accept CUDA.

If you qualify CUDA, create a new config and run identity. Start with smoke, verify
the upstream checkpoint, inspect the data, add a bounded device probe, and record
all results. Do not describe CUDA as supported end to end until downstream
behavior is also implemented and tested.

One implementation detail matters: foundation DataLoader pinning checks the
configured device string. If `device="auto"` resolves to CUDA, pinning remains
off. Use explicit `device="cuda"` when validating that code path.

## Understand precision and parallelism limits

The implemented pipeline uses:

- Float32 model and training tensors.
- One process and one device.
- No distributed data parallelism.
- No mixed-precision training.
- No `torch.compile`.
- No remote experiment tracker.

Safetensors export preserves model-state dtypes. Downstream embeddings are stored
as bounded float16 shards according to config.

## Verify setup in stages

Use the cheapest evidence first:

1. Confirm tool versions.
2. Run `just sync` only with installation authority.
3. Run `just smoke`.
4. Run `just downstream-smoke`.
5. Run `just verify-upstream`.
6. Run `just inspect-data <config>`.
7. Run the device-specific probe when one exists.

Do not run heavy checks alongside a resource-sensitive production job. See
[Protect an active run](workflow.md#protect-an-active-run).

## Setup limitations

- No automated market-data acquisition exists.
- Production paths are machine-specific examples.
- `uv`, `just`, Git, and driver versions are not pinned.
- The first official-weight fetch depends on Hugging Face availability or cache.
- CPU and CUDA production timing and memory requirements are unknown.
- Upstream license declarations conflict, so weights cannot be redistributed.

## Related documentation

- [Why the Mantis family is used](mantis-family.md)
- [System architecture](architecture.md)
- [End-to-end workflow](workflow.md)
- [Troubleshooting](troubleshooting.md)
