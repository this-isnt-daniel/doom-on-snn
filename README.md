# doom-on-snn

Spiking neural network (SNN) policies for the ViZDoom `doom_basic` scenario. The project compares two ways of
getting a spiking policy:

1. **ANN → SNN conversion**: train a conventional CNN, then convert it with threshold balancing.
2. **Direct SNN training**: train a spiking network from scratch with RL and surrogate gradients.

All notebooks are meant to run on **Google Colab or Kaggle (Linux, GPU)**, not native Windows.

## Pipeline at a glance

```
doom_samplefactory_pipeline_v4  ->  doom_plaincnn_sanity_check  ->  (converted SNN, same notebook)
        (train ANN)                  (extract + verify + convert)

doom_snn_direct_training (3)   (train SNN directly, compare against the above)
```

## Notebooks

### `doom_samplefactory_pipeline_v4.ipynb`: ANN baseline training
Trains the conversion-friendly ANN with [Sample Factory](https://github.com/alex-petrenko/sample-factory).

- Installs system dependencies, Sample Factory and ViZDoom; forces a CUDA PyTorch build.
- Applies the PyTorch 2.6 `torch.load(weights_only=False)` fix.
- Registers the ViZDoom envs/encoder and trains on `doom_basic` for 2M env steps with
  `--nonlinearity=relu --use_rnn=False` (plain feedforward CNN, no GRU, no frame stacking).
- Inspects the encoder first to confirm there is **no BatchNorm** (which would block conversion).
- Evaluates over 50 deterministic episodes, shows the replay video, and backs up the run to Drive.
- Architecture: Conv(3→32, k8 s4) → Conv(32→64, k4 s2) → Conv(64→128, k3 s2) → FC(2304→512) → 4-way action head.
- Note: the results section at the end still has "fill in" placeholders.

### `doom_plaincnn_sanity_check.ipynb`: weight extraction and the converted SNN
Two parts.

**Part 1: PlainCNN extraction**
- Rebuilds the trained `conversion_friendly_v1` checkpoint as a plain PyTorch `PlainCNN`, with no Sample Factory
  model classes. The real `ObservationNormalizer` is kept for preprocessing.
- Verifies the extraction reproduces the baseline reward (~0.76–0.80) and that weights match the checkpoint exactly.

**Part 2: ANN → SNN conversion (threshold balancing)**
- Collects calibration frames (clean and epsilon-noisy episodes), split into calibration and test sets.
- Measures per-layer activation scales (percentile-based λ), rescales weights and biases, and swaps ReLU for
  integrate-and-fire neurons (soft reset, half-threshold initial membrane).
- Offline checks: large-T logit match with the ANN, action agreement versus T, and a percentile sweep (99.99 chosen).
- In-game evaluation over 300 episodes at several T; saves `snn_converted_v2.pt` and per-episode rewards.
- Integrity checks (uniform per-layer scaling, saved file equals in-memory model) and spike/MAC/energy accounting.

### `doom_snn_direct_training (3).ipynb`: directly trained SNN
Trains a spiking policy from scratch with Sample Factory and surrogate gradients, for comparison with the converted SNN.

- Same layer sizes as the ANN; every hidden layer is integrate-and-fire (threshold 1, soft reset, v_init 0.5,
  analog input each step) run for `T` steps per frame, with a triangular surrogate gradient.
- State resets every frame (no recurrence, no frame stacking). The 512-layer's firing rate feeds the action and value heads.
- Settings cell controls `T` (default 8), seed, and a `SMOKE_TEST` mode (100k steps).
- Verifies the standalone SNN's logits match Sample Factory's model, then evaluates over 300 episodes at
  T = 2, 4, 8, 16, 32 using the same protocol as the converted SNN.
- Estimates spikes, operations and energy (45 nm: MAC 4.6 pJ, accumulate 0.9 pJ), compares with the ANN and the
  converted SNN, and saves results to Drive.

## Other files

- `conversion_friendly_v1/`: trained ANN run: checkpoints (best reward 0.822), `config.json`, `sf_log.txt`, `replay.mp4`.
- `nb_dump.txt`: plain-text dump of notebook code/markdown (no outputs), for quick reading and diffing.

## Results summary (300 episodes, `doom_basic`, single seed)

| Policy | Reward | Below 0 | Est. energy / decision |
|---|---|---|---|
| ANN | 0.786 ± 0.118 | 1 | ~41 µJ |
| Converted SNN T=32 | 0.798 ± 0.115 | 0 | ~33 µJ |
| Converted SNN T=64 | 0.791 ± 0.110 | 0 | ~51 µJ |
| Direct SNN T=8 (trained) | 0.804 ± 0.088 | 0 | ~20 µJ |
| Direct SNN T=2 | 0.822 ± 0.090 | 0 | n/a |

The directly trained SNN matches the ANN at much smaller T than the converted SNN (T=8 versus T=32).
Energy numbers are order-of-magnitude estimates that ignore memory access.


