# Defend the center experiments

Two independent Sample Factory pipelines for the standard
`doom_defend_the_center` scenario:

- [ANN baseline](defend_the_center_ann.ipynb): a feedforward ReLU CNN using Sample
  Factory's ViZDoom encoder, trained from scratch.
- [Direct LIF SNN](defend_the_center_lif_snn.ipynb): the existing custom spiking
  encoder architecture with beta=0.9 and T=8, trained from scratch using PPO and
  surrogate gradients. It does not use the ANN checkpoint.

The action space is `Tuple(Discrete(3), Discrete(2))`: turning (none, left, right)
and attack (off, on). Five logits describe six possible combinations. This is not
the four-action flat variant. Both notebooks validate the installed action space
and use the full Sample Factory model for evaluation, including its trained
normalizer and decoder.

## Run on the server

Push and pull this folder, open Jupyter from the repository, and select a fresh
GPU kernel for each notebook. Run one notebook/workload at a time. Existing Python,
CUDA, Sample Factory, ViZDoom and Gymnasium packages are reused; no installs or
administrator commands run. Jupyter must run inside the cluster allocation.

1. Keep `RUN_MODE="fresh"` and `SMOKE_TEST=True` initially. The smoke test trains
   100k steps and evaluates 10 episodes. LIF calibration still collects 800 frames.
2. Check environment creation, training status, checkpoint reload and evaluation.
3. For full experiments set `SMOKE_TEST=False` in both notebooks: 8M steps and 300
   evaluation episodes. This is a proposed matched budget, not a convergence claim.
4. Keep a distinct `RUN_TAG` for every independent run. Fresh mode refuses an
   existing experiment directory. Resume/evaluate requires its saved config.
5. To resume or evaluate, select the original seed/tag/smoke setting so the same
   experiment path is selected, then set `RUN_MODE` appropriately. LIF reads saved
   gains/beta/T instead of recalibrating. Inspect resolved logs/config when changing
   a resumed training target: saved settings can override requested flags.

Default artifacts live under `defend the center/outputs` and are ignored by Git.
Set `DEFEND_OUTPUT_ROOT` before launching Jupyter to use persistent cluster storage.
Worker count respects Slurm's CPU count or affinity, capped at eight; optionally
set `DOOM_NUM_WORKERS`. Neither variable requests a scheduler allocation.

Each experiment retains its checkpoints/config/logs, LIF calibration artifact,
episode reward array, and JSON evaluation report containing package versions,
GPU, checkpoint hash and resolved training configuration. The latest regular
checkpoint is evaluated, not necessarily the best-reward checkpoint. Re-evaluation
of the same experiment currently overwrites its evaluation report/reward array.

## Best-checkpoint replay

The final replay cell in each notebook separately loads `best_*.pth`, selecting
the highest reward encoded in its filename. This is Sample Factory's best recorded
training reward, not a claim of best held-out performance. If no best snapshot
exists yet, the cell asks you to continue training rather than silently using latest.

It records three deterministic episodes (seeds 20000-20002), saves MP4s under
`outputs/replays/<EXPERIMENT>/` and embeds them in Jupyter. The metadata records
the checkpoint hash, steps, seeds and returns. Those seeds are now demonstration
seeds; use a different range for any untouched final evaluation set. Repeated replay
overwrites the same seed-specific files. This does not replace the earlier latest-
checkpoint evaluation or its report.

ffmpeg with libx264 support must be available in the Jupyter process's PATH; load
your cluster module before starting Jupyter if needed. Videos stream RGB frames
into ffmpeg with argument-list invocation, supporting spaces in paths and avoiding
whole-video RAM buffering. Encoding logs are saved beside each video. Videos contain
one frame per environment decision at 35/frame_skip FPS, not every underlying tic.
Local checks use a simulated renderer/encoder; real ViZDoom replay must be checked
on the server.

## Comparison protocol

After checkpoint loading, the ANN notebook prints the full architecture, traces
convolution/linear shapes and reports module-level BatchNorm, non-ReLU activation,
max-pooling and recurrence concerns for future conversion. It saves
`results/<EXPERIMENT>/ann_architecture.json`. This is an inspection aid, not proof
that functional operations, preprocessing or tuple-action extraction are correct.

The LIF notebook measures activity on up to 500 gameplay observations across at
most ten diagnostic episodes starting at seed 30000. It reports each layer's
neuron count, spikes per observation, firing rate and zero-spike-event fraction,
and saves `results/<EXPERIMENT>/lif_activity.json`. Rate divides spikes by
observations times neurons times saved training T. Sparsity here means silent
neuron-time slots, not neurons that never fire. The diagnostic uses the latest
loaded checkpoint; the later replay independently uses best training reward.
It adds gameplay and a second trunk pass per observation but does not update weights.

Both models use 72x128 RGB images, frame skip 4, no RNN, learning rate 1e-4, batch
size 1024 and the same evaluation episode seeds (10000 onward). Repeat full
training with seeds 0, 1 and 2 before claiming a consistent advantage. Use a separate
evaluation seed range for final reporting after tuning. Seeding does not guarantee
identical asynchronous training or software behavior across machines.

Report return and decisions survived; these notebooks do not assume return equals
kill count or reuse doom_basic reward thresholds. Decisions are environment steps,
not raw game tics or seconds. No energy estimate or ANN-to-SNN conversion is included.
The LIF membrane resets each observation, so it has no cross-frame membrane memory.

The trusted-checkpoint environment setting is applied before workers launch to
handle PyTorch's restricted loading default. Only use trusted checkpoints in these
kernels. An existing IF/basic checkpoint is not compatible with the new experiment.

## Validation and sources

Local verification covers code syntax, tuple-action decoding, real PyTorch LIF
forward/backward computation and five-logit checkpoint reconstruction. Full Sample
Factory/ViZDoom training is not validated locally and must pass the server smoke
test first. These notebooks use APIs from the existing pipeline; installed package
differences may still need adjustment.

- [Scenario registry](https://github.com/alex-petrenko/sample-factory/blob/master/sf_examples/vizdoom/doom/doom_utils.py)
- [Turn-and-attack action space](https://github.com/alex-petrenko/sample-factory/blob/master/sf_examples/vizdoom/doom/action_space.py)
- [Tuple policy distribution](https://github.com/alex-petrenko/sample-factory/blob/master/sample_factory/algo/utils/action_distributions.py)
