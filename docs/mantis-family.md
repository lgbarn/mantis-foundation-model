# Why the Mantis family is used

This guide explains Mantis, Mantis+, and MantisV2 for readers who are new to
machine learning. It also explains why this repository uses a hybrid
foundation-model workflow instead of relying only on a traditional feature
engineering pipeline.

## Start with the basic idea

A time series is a sequence of measurements in time order. A futures bar is one
example. It commonly contains:

- Open is the first traded price in the bar.
- High is the highest traded price.
- Low is the lowest traded price.
- Close is the last traded price.
- Volume is the number of traded contracts.

These 5 values are abbreviated as OHLCV.

A traditional supervised model receives inputs selected by a person and learns
to predict a label. For trading, a person might provide returns, moving averages,
Average True Range (ATR), volume changes, or indicator states. The model learns
only from that feature vocabulary.

A foundation model is pretrained on a broad task before it is adapted to a
specific task. MantisV2 is a small time-series classification foundation encoder.
It converts a sequence into an embedding: a vector of numbers that summarizes
patterns in the sequence. A downstream model can then learn from the embedding.

MantisV2 is not a pretrained trader. It does not know futures contracts, market
sessions, execution costs, or account rules. Its pretraining provides a general
time-series representation that this repository tests and adapts.

## Compare the model family

| Model | Backbone | Upstream pretraining | Approximate size | Status in this repository |
| --- | --- | --- | ---: | --- |
| Mantis | MantisV1 | Real time-series collections; the published corpus count is inconsistent across paper versions | 8.1M parameters | Not implemented |
| Mantis+ | MantisV1 | 2M synthetic CauKer sequences for 200 epochs | 8.1M parameters | Not implemented |
| MantisV2 | Refined MantisV2 | The same 2M synthetic CauKer sequences for 200 epochs | 4.19M parameters | Only active family |

Mantis+ changes the pretraining data and duration while retaining the MantisV1
architecture. MantisV2 changes the architecture as well. None of the 3 models is
pretrained on trading data.

## What Mantis does

Mantis was designed for time-series classification. Most earlier time-series
foundation models focused on forecasting or reconstruction. Mantis instead uses
contrastive learning: it creates 2 altered views of one sequence, moves their
representations closer together, and moves representations of different
sequences apart.

The original design:

1. Resizes or pads a sequence to 512 values.
2. Processes each input channel independently.
3. Forms 32 tokens from the normalized signal, first differences, and local
   statistics.
4. Passes the tokens through a 6-layer Transformer.
5. Produces an embedding for downstream classification.

The original paper reports an 8M-parameter model trained on real-world time
series. It includes activity, biomedical, sensor, and benchmark datasets. It does
not include documented market bars or realistic trading evaluation.

### An unresolved Mantis corpus discrepancy

The first Mantis paper version reports 7M pretraining examples. The later
MantisV2 paper describes the original Mantis corpus as 1.89M examples and says it
overlapped some benchmark training sets. The sources do not provide enough
information to reconcile those counts. This repository preserves that
uncertainty instead of choosing a convenient number.

## What Mantis+ changes

Mantis+ keeps the MantisV1 architecture and replaces the real-data pretraining
corpus with 2M synthetic CauKer sequences. CauKer composes mathematical processes
that create trends, seasonality, nonlinear relationships, and other time-series
shapes.

This change has 2 benefits:

- Synthetic pretraining avoids direct overlap with standard real-data benchmark
  collections.
- Comparing Mantis with Mantis+ isolates the effect of the pretraining corpus
  without changing the architecture.

Mantis+ does not gain financial knowledge from this synthetic data. CauKer does
not model contract rolls, session boundaries, tick sizes, spread, slippage,
order flow, or trading costs.

## What MantisV2 changes

MantisV2 retains the main tokenizer and 6-layer encoder idea but refines the
architecture:

- The convolution kernel grows from 17 to 41.
- Attention uses standard 32-dimensional heads instead of the oversized V1
  projection.
- Transformer blocks use SwiGLU, RMSNorm, and rotary positional encoding.
- Full model size falls from about 8.1M to about 4.19M parameters.
- Frozen feature extraction can use a useful intermediate layer and combined
  output token.

The smaller model is important for this repository. It can be fine-tuned locally
on Apple Silicon while retaining a pretrained Transformer representation.

## What zero-shot means here

In the Mantis papers, zero-shot feature extraction does not mean label-free
prediction. It means the encoder remains frozen when it processes a new dataset.
A conventional classifier still trains for the new task. It may be Random Forest
or logistic regression. It still needs labeled examples.

There are 3 common adaptation modes:

| Mode | What changes | Benefit | Risk |
| --- | --- | --- | --- |
| Frozen features | Only a downstream classifier trains | Low compute and stable embeddings | Domain knowledge may remain weak |
| Partial adaptation | A head or adapter and head train | More domain flexibility with limited compute | More moving parts |
| Full fine-tuning | Encoder and prediction heads train | Maximum domain adaptation | More compute and overfitting risk |

This repository uses full fine-tuning for the NextLeg stage, then freezes the
validated result for downstream embedding extraction.

## Why MantisV2 is the active choice

MantisV2 is active for specific reasons:

1. It is the newest and smallest architecture in the family.
2. Its synthetic-only pretraining has a cleaner benchmark separation story than
   the original Mantis corpus.
3. Its paper, source commit, package version, released weights, and weight digest
   can be pinned independently.
4. Its size is practical for local Apple MPS fine-tuning.
5. Building one complete family workflow avoids speculative abstractions across
   model versions.
6. The legacy Mantis checkpoint in the related repository uses V1 parameter keys
   and is not compatible with MantisV2.

Mantis and Mantis+ remain possible controlled baselines. Each one needs its own
source and weight pins. Each one also needs an adapter, configs, tests, artifacts,
and a design decision. Do not create them by copying the V2 directory.

## What this repository actually trains

This repository does not reproduce MantisV2 foundation pretraining. It begins
with the official pretrained MantisV2 weights and performs supervised,
futures-specific NextLeg fine-tuning.

For each legal futures anchor:

1. The pipeline obtains 64, 100, 150, or 200 historical OHLCV bars.
2. It calculates normalization statistics from the historical context only.
3. It resizes each of the 5 channels to the model's 512-value input.
4. MantisV2 encodes each channel independently.
5. The local model concatenates the 5 channel embeddings.
6. One head predicts standardized candle changes at 5, 10, 20, and 25 bars.
7. Another head predicts the durations of 2 future pivot legs.

Validation selects the best checkpoint. Export must then pass its parity check.
The next stage freezes the encoder. It builds 1-minute, 3-minute, and 15-minute
embeddings. A logistic classifier trains in purged walk-forward folds. Selected
candidates then enter the rule-based simulator.

## Compare this with a traditional ML pipeline

| Question | Traditional feature pipeline | MantisV2 hybrid pipeline |
| --- | --- | --- |
| Who defines features? | A person chooses indicators, lags, and interactions | The encoder learns representations from sequence windows |
| Interpretability | Usually higher | Lower inside the encoder; higher in the logistic head |
| Compute cost | Usually lower | Higher during fine-tuning and embedding extraction |
| Label efficiency | Must learn from task labels alone | Starts from synthetic contrastive pretraining |
| Cross-market reuse | Features often need redesign | One encoder interface can process many streams |
| Reproducibility surface | Data, features, model, and split | Adds upstream weights, preprocessing, device, and encoder state |
| Leakage risk | High if features or splits use future data | Still high; foundation models do not remove it |
| Baseline value | Essential | Must prove improvement over this baseline |

The correct claim is that MantisV2 reduces dependence on manual features. It does
not eliminate traditional ML. The current workflow keeps traditional components
where they are useful:

- The configured Supertrend or Trend Magic profile defines a causal directional candidate.
- Logistic regression produces an inspectable probability.
- Validation owns the selection threshold.
- Walk-forward testing measures time-local generalization.
- The simulator owns execution, costs, and account rules.
- Handcrafted and statistical baselines remain required comparisons.

## Problems the approach may solve

### Learn reusable representations

The encoder can learn local shapes and longer interactions without requiring a
person to enumerate every pattern as an indicator.

### Start with limited labels

Synthetic contrastive pretraining may provide a better starting point than random
initialization when labeled market outcomes are noisy or scarce. This remains a
hypothesis that must be tested against scratch training.

### Reuse one interface across streams

The model accepts a fixed 512-value input and can process many symbols and
timeframes. The data pipeline still keeps streams separate until causal alignment
is complete.

### Bound model complexity

MantisV2 is much smaller than many forecasting foundation models. The downstream
classifier is deliberately simple, which makes probability fitting and threshold
selection easier to inspect.

### Reproduce long workflows

The local pipeline adds provenance, resume state, validation-owned selection, and
export parity beyond the upstream training interface.

## Challenges the approach does not remove

### Classification and forecasting are different tasks

MantisV2 was pretrained for classification representations, not numerical market
forecasting. The local objective must teach magnitude and horizon behavior.

### The model has no native timeframe meaning

MantisV2 sees ordered values. It does not know that a sample represents 1 minute,
15 minutes, a session gap, or a holiday. Configs and causal alignment carry that
meaning.

### Synthetic data differs from markets

Synthetic shapes do not reproduce market microstructure, news, rolls, tick size,
liquidity, or costs. Domain fine-tuning and untouched evaluation are mandatory.

### Channels are encoded independently

Attention does not jointly model OHLCV channels or timeframes inside the encoder.
Cross-channel relationships are learned after embeddings are concatenated.

### Normalization can remove scale

Normalization improves transfer but can hide economically meaningful absolute
levels. Interpolation also changes the density of short contexts.

### Embeddings are opaque

An embedding dimension is harder to explain than ATR or a moving average.
Ablations and traditional baselines are required.

### Overfitting and regime change remain

Closely overlapping windows are not independent observations. Full fine-tuning
can learn behavior from one time period. Purging and embargo reduce this risk.
Early stopping and walk-forward tests also help. A sealed holdout provides the
last check, but none of these steps can remove all risk.

### Hardware can change numeric behavior

CPU, Apple MPS, and NVIDIA CUDA can differ in speed and memory. They can also
support different kernels. Their numeric results may differ. A checkpoint is
bound to its config. Do not silently move it between production recipes.

### Better prediction does not imply profit

Regression loss, log loss, Brier score, and area under a curve measure model
behavior. They do not include spread, fees, bad fills, capacity, or account
rules. The simulator is a separate evidence stage.

## Evidence boundaries

The Mantis papers report time-series classification results. Their data includes
UCR, UEA, human-activity, and EEG sets. They do not show profitable trading. They
also do not show futures transfer, realistic fills, or cost robustness.

The repository must compare the hybrid approach with traditional features,
simple statistical models, scratch training, and frozen upstream embeddings. It
must reserve final claims for the sealed holdout. Until then:

- Do not call MantisV2 a proven trading model.
- Do not call local NextLeg training foundation pretraining.
- Do not treat zero-shot as label-free prediction.
- Do not infer profitability from validation loss.
- Do not publish derived weights while the upstream license conflict remains.

## Primary references

- [Mantis paper, arXiv:2502.15637v1](https://arxiv.org/abs/2502.15637v1).
- [MantisV2 paper, arXiv:2602.17868](https://arxiv.org/abs/2602.17868).
- [CauKer paper, arXiv:2508.02879](https://arxiv.org/abs/2508.02879).
- [Pinned upstream source](https://github.com/vfeofanov/mantis/tree/0c94f8ceb9f1d1421dd292ed917090df8c31605b).
- [Pinned MantisV2 weights](https://huggingface.co/paris-noah/MantisV2/tree/99fe0f548960e272fbfa4b82fd9b5b5956779dfd).
- [Local pinning decision](adr/0001-pin-and-isolate-mantis-v2.md).

## Glossary

| Term | Plain-language meaning |
| --- | --- |
| Anchor | The bar position at which one training example is formed |
| Bar | OHLCV values collected over one interval |
| Batch | A group of examples processed before one optimizer update |
| Checkpoint | Saved model and training state used to resume or evaluate |
| Embedding | A learned numeric summary of a sequence |
| Encoder | A model that converts an input sequence into a representation |
| Epoch | One configured training cycle; here it is capped and is not a full corpus pass |
| Feature | A numeric model input |
| Fine-tuning | Updating pretrained weights for a new task |
| Foundation model | A model pretrained broadly and adapted to later tasks |
| Holdout | Final data reserved from all routine tuning and selection |
| Horizon | How far into the future a target refers |
| Label or target | The outcome a supervised model learns to predict |
| Loss | A numeric measure the optimizer tries to reduce |
| MPS | Apple's Metal Performance Shaders acceleration backend |
| CUDA | NVIDIA's GPU computing platform |
| OHLCV | Open, high, low, close, and volume |
| Pretraining | Training before adaptation to a specific downstream task |
| Purge | Removing examples whose information spans cross a split boundary |
| Timeframe | The duration represented by one bar |
| Transformer | A neural architecture that models relationships among tokens |
| Walk-forward | Repeated chronological train, validation, and test evaluation |
