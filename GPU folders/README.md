# Linux GPU server notebooks

These three copies use the existing university server environment. The original
notebooks and repository checkpoint are unchanged. No notebook runs apt, installs
packages, replaces CUDA PyTorch, or mounts Google Drive.

## Run after pushing and pulling

On the server, stop active notebook kernels before pulling changes, and save any
server edits first. A Git pull can fail if it would overwrite local edits.

```bash
cd ~/doom-on-snn
git pull
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

Keep your working SSH tunnel open. In the browser, open `GPU folders`, choose a
notebook, and use a fresh kernel. Run from the top; do not skip the first setup cell.
Your existing Python 3 kernel is suitable if its GPU/package checks still pass.
Use a separate kernel for each notebook because their Sample Factory registrations
and model definitions differ. Prefer running one workload at a time.

- `doom_plaincnn_sanity_check.ipynb`: start here to use the trained checkpoint
  already in `conversion_friendly_v1`. No baseline retraining required. The server
  copy uses the normalizer's returned dictionary for evaluation and calibration;
  this corrects ignored return values in the original notebook. Scores may differ
  from runs that used those original cells.
- `doom_samplefactory_pipeline_v4.ipynb`: train a new ANN baseline for 2M steps.
  Training goes to `outputs/train_dir/conversion_friendly_v1`; it does not replace
  the committed checkpoint. To convert this new run, change `EXPERIMENT_DIR` and
  the checkpoint filename in the PlainCNN notebook. Video recording is disabled
  by default; enable `--save_video` only when ffmpeg is installed.
- `doom_snn_direct_training (3).ipynb`: independent SNN training, 2M steps by
  default. Set `SMOKE_TEST = True` in its settings cell for a 100k-step check first.
  This shortens training only; later calibration/evaluation cells still perform
  their configured full workloads. The notebook writes `snn_direct.py` into this
  folder for its importable encoder implementation.

## Output paths and resource controls

Default persistent outputs are under `GPU folders/outputs/`, ignored by Git:

```text
outputs/environment.json
outputs/train_dir/<experiment>/       # checkpoints, config and logs
outputs/converted_snn/                # calibration frames, converted weights, rewards
outputs/direct_snn_results/           # direct SNN metrics and reward arrays
```

To use cluster storage or limit workers, set variables **before starting Jupyter**:

```bash
export DOOM_OUTPUT_ROOT=/your/persistent/storage/doom-results
export DOOM_NUM_WORKERS=4
```

The default worker limit uses `SLURM_CPUS_PER_TASK` if set, otherwise the process CPU
affinity. It reserves two CPU slots when possible and caps workers at eight. This
is a starting point, not a scheduler allocation; workers, learner and Jupyter all
need CPU resources. Sample Factory's own CPU affinity pinning is disabled so the
cluster allocation controls placement. Training explicitly requests GPU execution.

## Potential issues

- **Cluster allocations:** these notebooks do not request a GPU job. Launch Jupyter
  and its kernel within the allocation required by your department. Its wall-time
  limit still applies. Do not use a login/gateway node for training.
- **Environment compatibility:** imports and a CUDA calculation are checked, but
  Sample Factory/ViZDoom API compatibility still needs actual server execution.
  Your working PyTorch build is retained. `environment.json` records versions;
  it is not a dependency lockfile. No package upgrades are needed just to try this.
- **Native libraries:** if ViZDoom environment creation reports missing shared
  libraries, use the cluster's modules/container or ask its admin. Import success
  alone does not prove every runtime library is present. Simulation is headless.
- **Checkpoint trust:** Sample Factory checkpoints contain NumPy metadata. A
  rerun-safe loader wrapper defaults to `weights_only=False` for trusted project
  checkpoints, including Sample Factory's internal loads. Explicit caller settings
  are preserved. Do not open untrusted checkpoint files in these kernels.
- **Existing runs:** reusing an experiment name can resume an existing Sample
  Factory run. Change the experiment name or output root for an independent run.
  Reusing the conversion output folder also reuses cached calibration frames;
  select a new output root when changing source checkpoints or preprocessing.
- **Memory and speed:** calibration frames consume substantial host RAM, and
  large-T SNN sweeps are expensive. Reduce direct SNN `BATCH_SIZE` if CUDA memory
  runs out. A GPU does not make environment simulation or single-frame evaluation
  entirely GPU-bound.
- **Disconnections:** closing the browser does not normally stop the kernel, but
  losing the Jupyter process or cluster job does. Use your cluster's supported
  persistent session/job method. Keep backups of results; Git ignores outputs.
- **Git sync:** new files must be added, committed and pushed on the laptop before
  pulling on the server. Git does not automatically synchronize live notebook
  kernels, unsaved browser changes, installed packages or ignored result files.

## Validation performed locally

The copies were checked for notebook structure, Python/IPython cell syntax, removed
Colab commands, clean execution state, output path consistency and loader rerun
behavior. Full ViZDoom training/evaluation has not been run on the university GPU.
