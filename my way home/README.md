# My way home: ANN and snnTorch navigation policies

For the separate random Explore-Go-inspired ANN experiment, see
[EXPLORE_GO.md](EXPLORE_GO.md) and
[my_way_home_ann_explore_go.ipynb](my_way_home_ann_explore_go.ipynb).
It includes three matched comparison arms and requires native smoke validation
before full training.

Two independent notebooks:

- [my_way_home_ann.ipynb](my_way_home_ann.ipynb): Sample Factory PPO with a conventional
  ReLU CNN and analog action head.
- [my_way_home_snntorch.ipynb](my_way_home_snntorch.ipynb): Sample Factory PPO with
  rate-coded input spikes, snnTorch LIF hidden layers, **spiking action outputs**,
  and fixed spike-count/rate decoding. No trained ANN checkpoint is needed.

Keep these helpers in the same folder when uploading or pulling on the server:

```text
my way home/
    my_way_home_ann.ipynb
    my_way_home_snntorch.ipynb
    home_models.py
    home_env.py
    home_worker.py
```

## Start with a smoke test

Use your existing Linux GPU environment with PyTorch, Sample Factory, ViZDoom,
Gymnasium, NumPy and Matplotlib. Only the SNN additionally requires snnTorch.
If needed, use `%pip install snntorch` in the intended environment, restart the
kernel, and keep the existing working CUDA PyTorch build. Tested neuron API:
snntorch 1.0.0. Actual server integration may vary with installed package versions.

Start Jupyter from the repository or this folder. Open one notebook in a fresh
kernel and run the settings/plan cells first. Defaults are:

- `ENABLE_TRAINING=False`: review before launching expensive computation.
- `SMOKE_TEST=True`: 100k requested training steps and ten evaluation episodes.
- Training seed 0, frame skip 4, one RGB image per observation, learning rate 1e-4,
  batch size 1024. SNN uses T=8, beta=.9 and fixed decoder scale 8.

Set `ENABLE_TRAINING=True`, rerun the settings, and launch the train cell. Then
run evaluation, charts and replay. Full mode (`SMOKE_TEST=False`) requests 8M steps
and 300 evaluation episodes; this is a starting budget, not a convergence claim.
Keep a distinct `RUN_TAG` for each independent run. Seed 1/2 runs are independent
training attempts, with the same evaluation seed list.

Fresh training refuses an existing experiment directory. To continue a run, use
its original experiment name and `RUN_MODE="resume"`; it reads saved settings and
preserves the initial calibration. Resume explicitly applies the requested total
training target, but architecture/optimizer defaults otherwise come from the saved
configuration. For evaluation only use `RUN_MODE="evaluate"` and leave training
disabled. Normal evaluation selects `CHECKPOINT_KIND`, while replay always selects
the highest recorded training-reward snapshot. Missing checkpoints cause explicit
errors; best training reward is not guaranteed to maximize held-out success.

The notebook controller does not initialize CUDA. Isolated GPU workers execute
training/evaluation/replay sequentially. A log path is printed every 30 seconds;
errors print the end of that log. On interruption, the controller terminates only
its own worker process group. Check for surviving processes before restarting an
interrupted training attempt. Only load trusted project checkpoints: worker loading
allows the metadata needed by Sample Factory.

## Exact SNN design

```text
Stacked RGB pixels in [0,1]
  → deterministic rate-coded binary input spikes over T steps
  → Conv(3*stack→32, k8/s4) + snnTorch LIF
  → Conv(32→64, k4/s2) + snnTorch LIF
  → Conv(64→128, k3/s2) + snnTorch LIF
  → Linear(2304→512) + snnTorch LIF
  → Linear(512→6) + snnTorch LIF
  → fixed scale × six output firing rates
  → two categorical action distributions (turning and movement)
```

Synaptic linear/convolution operators compute currents; each policy neural stage
ends in a spiking neuron. There is **no learned neural readout after output spikes**.
PPO samples from softmax(scaled rates) during training. Evaluation chooses the
largest count per three-neuron group. Ties pick the first index; silent groups and
ties are reported. Scaling is fixed and saved, not an extra trained layer.

Rate encoding uses an accumulator initialized to .5: add pixel intensity each step,
emit a binary spike at 1, and subtract 1. Across T steps the count approximates
pixel*T with at most half a spike rounding error. It encodes intensity as firing
rate; it is not Poisson encoding or a stochastic augmentation. Determinism prevents
fresh encoder noise from changing PPO likelihoods when the same observation is
recomputed. More T gives finer rate resolution but costs more computation. The
input encoder operates on scaled pixels, without running observation centering.
The ANN uses the same pixel scaling for this controlled comparison.

All LIF layers use beta .9, threshold 1, initial membrane .5, immediate subtractive
reset and a triangular surrogate. Native snnTorch reset gradients are used; this
is not claimed to reproduce the earlier custom IF/LIF implementation exactly.
States reset for each observation. `STACK=2/4` supplies recent images but is not
a recurrent SNN. No conventional GRU/LSTM is used.

The **training-only critic is analog**, reading 512 hidden firing rates to estimate
return. Weights, currents and voltages remain floating point. This is a spiking
action policy with rate-coded input/output and a conventional PPO value estimator;
it is not neuromorphic hardware or a claim of measured energy savings.

Five fixed gains (four hidden, one output) are calibrated on random gameplay using
the actual initialized actor weights. Gain search uses a bounded subset of saved
calibration frames to control GPU memory. Initial state and gains are saved and
loaded into training workers. Rates need not exactly match the target; silence,
ties and learning curves need inspection. Calibration, encoding and count decoding
are new experimental choices rather than proven improvements over the old model.

## Scenario controls and reward

Sample Factory declares `Tuple(Discrete(3),Discrete(3))`: turn none/left/right and
move none/forward/backward. This allows nine combinations, using six logits.
The released Sample Factory 2.1.1 bundled config was checked locally and exposes
exactly those four buttons. ViZDoom's generic my_way_home config can instead expose
forward/strafe controls, so these notebooks deliberately use Sample Factory's asset
and validate the actual buttons before training. A mismatch fails rather than
silently relabeling controls. Do not substitute generic scenario cfg files.

Native my-way-home rewards are retained: +1 when collecting the goal vest and
-0.0001 per game tic. The default engine timeout is 2100 tics. Success is detected
from a reward >.5, consistent with that goal event; the metric must be revised if
scenario scripts/rewards are changed. Timeouts are detected from engine time,
because Sample Factory may mark a native timeout as terminated rather than
Gymnasium-truncated. The installed scenario path/hash/reward/timeout are recorded.

## Evaluation and replay

Evaluation explicitly seeds both the wrapper and an already-initialized Doom
engine before starting each episode, covering Sample Factory versions where
reset(seed=...) alone only updates the wrapper's seed variable.
Evaluation plays the same episode seeds (10000 onward) for ANN and SNN, using
deterministic per-factor argmax actions. Reports include original return, goal
success rate, timeout rate, decisions, time-to-goal among successes, and synchronized
batch-1 inference latency. SNN diagnostics additionally record input/hidden/output
firing rates, output ties and complete silence per action group. Small-T ties can
matter and are not silently resolved using analog membrane voltage.

Replay uses the best recorded training checkpoint and demonstration seeds
20000-20002. ffmpeg with libx264 must be in PATH. RGB frames stream into MP4s using
argument lists, which handle paths with spaces. Videos play inside Jupyter and
have matching per-decision action CSVs. Short smoke tests may not produce a best
checkpoint; continue training or deliberately choose an existing completed run.

Default output root is `my way home/outputs`; override `HOME_DOOM_OUTPUT_ROOT`
before starting Jupyter to use persistent cluster storage. Git ignores outputs.

```text
outputs/
    controller/<experiment>/        # stage jobs and worker logs
    train_dir/<experiment>/         # checkpoints, config, training logs
    results/<experiment>/
        calibration.pt             # SNN only
        initial_state.pt           # SNN worker initialization; preserve this
        scenario.json
        training_sessions.json
        evaluation_latest/         # or evaluation_best
            episodes.csv
            summary.json
        replay_best/
            seed20000.mp4          # and other demo seeds
            seed20000.actions.csv
            summary.json
```

Evaluation/replay reruns overwrite the selected report/video paths; stage job/log
files retain unique timestamps. Checkpoint hashes identify exactly which policy was
tested. Keep initial_state.pt with the SNN config when moving runs. Multi-seed
conclusions require multiple trained models; use a separate untouched episode set
for final reporting after tuning. Sparse spikes do not establish GPU energy savings.

## Validation

Local checks with real snnTorch 1.0.0 cover binary/rate-correct encoding, calibration
including the spiking output, deterministic and batch-independent outputs, finite
gradients through all synaptic layers, no learned post-spike decoder, membrane-state
cleanup and checkpoint round-trips. Sample Factory integration is checked against
its released source and tested with adapters locally; full multiprocessing PPO,
ViZDoom gameplay and video rendering require the Linux server smoke test.

References: [Sample Factory scenario/actions](https://github.com/alex-petrenko/sample-factory/blob/master/sf_examples/vizdoom/doom/doom_utils.py),
[snntorch LIF](https://snntorch.readthedocs.io/en/latest/snn.neurons_leaky.html),
[ViZDoom scenarios](https://vizdoom.farama.org/environments/default/).
