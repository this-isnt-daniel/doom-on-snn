# Random Explore-Go-inspired ANN baseline

Open [my_way_home_ann_explore_go.ipynb](my_way_home_ann_explore_go.ipynb).
This is a separate experiment; the existing ANN/SNN notebooks, CNN model and
Sample Factory PPO/GAE implementation have not been modified.

The [Explore-Go paper](https://arxiv.org/abs/2410.03565) motivates the two-phase
mechanism. This implementation uses a separate **random** explorer, not its
learned explorer, and makes no claim to reproduce its results. Random prefixes
are not guaranteed to improve navigation on the fixed bundled map.

Keep these new files alongside the existing home helpers:

```text
my_way_home_ann_explore_go.ipynb
explore_go_core.py
explore_go_sf.py
explore_go_worker.py
explore_go_control.py
test_explore_go.py
home_env.py
home_models.py
home_worker.py
```

Use the existing Linux GPU Python environment. This ANN notebook does not need
snnTorch. The implementation targets Sample Factory 2.1.1 APIs and fails visibly
if the training-statistics interface differs. Local verification used real
Gymnasium 1.4.0, PyTorch and the installed SF 2.1.1 GAE/recorder source; native
Linux ViZDoom/PPO runs must pass the notebook's separate smoke gate.

## Run order

1. Review settings and run the mechanism-test cell.
2. Set `ENABLE_SMOKE_TRAINING=True`, rerun settings, and run the short validation
   cell. It trains and evaluates **all three arms**, saves checkpoints and checks
   finite PPO metrics and interaction accounting. Smoke runs have separate names.
3. Inspect the logs/results. Only after the smoke certificate is created, enable
   `ENABLE_FULL_TRAINING=True`, rerun settings, and run the full training cell.
4. Run evaluation and chart cells. These always evaluate ordinary resets without
   any exploration prefix. Evaluation does not launch training automatically.

Defaults: four-frame stacks, frame skip 4, ANN learning rate 1e-4; three full
training seeds per arm. Full arms share identical PPO settings. The native smoke
uses fewer workers and a smaller PPO batch to verify the mechanism cheaply.
Training is sequential. Logs stream to the notebook and are saved on disk.

The three arms are zero-prefix baseline, random fixed K=50, and random K sampled
uniformly from integers 0 through 100. Explore-Go arms use ordinary-reset episodes
with probability 0.5. Each prefix is capped at 25% of the remaining decisions and
must leave a Go decision. Settings are editable in the notebook. Changing source
or the tested environment/exploration protocol requires fresh smoke validation.

## Boundaries and budgets

`ExploreGo` wraps **outside** `HomeImages`: its internal reset-prefix actions
update the real frame stack. The same live episode continues into Go, with its
remaining time intact. Prefix transitions are never returned to the sampler;
only Go actions/rewards/log-probabilities/values enter PPO. An exploration goal or
timeout is logged and retried with a bounded number of native resets. Prefix
success is not policy evaluation success.

PPO/GAE are unchanged. Mid-episode rollout boundaries bootstrap from the actual
next Go observation. Goals and the scenario's native finite-horizon timeout are
terminal; no value leaks into the next episode. An artificial external time-limit
truncation is rejected because this integration does not implement a separate
continuing-task terminal-observation bootstrap. The scenario remains the finite
2,100-tic task, with `value_bootstrap=False` and `with_vtrace=False` for every arm.
SF's extra random decorrelation steps are disabled for every arm.

`TOTAL_BUDGET` is a cumulative **Explore + Go** target. Its unit defaults to
`game_tics`, matching the usual frameskip-scaled SF budget interpretation; choose
`decisions` explicitly for a decision budget. Both actual counts are saved.
Terminal partial actions use the original native living reward/goal bonus to
recover their exact tic count, with a game-clock cross-check where available.
Changing reward schemes is unsupported and fails rather than silently guessing.
Reset initialization tics and separate evaluation work are not training actions.

Asynchronous workers can overshoot the target before stopping. Actual totals,
overshoot, Go-only learner counts, discarded terminal prefixes and wall time are
reported; equal requested budgets do not imply exactly equal realized totals.
The ledger also counts in-flight samples that never make it into an update.

## Storage and resuming

Workers update counters in shared RAM under a short lock. They perform no
per-action database writes or manager-server requests. The parent atomically
replaces `ledgers/<experiment>.json` approximately every 30 seconds and flushes it
at orderly shutdown. NFS can now hold these infrequent snapshots without SQLite
WAL locking. Checkpoint/log I/O and shared GPU load can still affect throughput.
Set `COUNTER_SAVE_SECONDS` in the notebook to change the save interval.

Forced termination, node failure or power loss can lose counts since the last
successful snapshot. Live budget checks still use the exact shared RAM counts;
on crash/resume the total can therefore undercount unsaved work. Retain snapshots
alongside model checkpoints, and do not claim crash-exact interaction accounting.
Only aggregate outcome counts are saved; per-reset database event rows are no
longer recorded. `training_metrics.jsonl` and evaluation outputs remain available.

Fresh mode refuses existing checkpoints/ledgers. To continue an interrupted run,
inspect its log, retain its name/settings, set `RUN_MODE="resume"`, and run just
that job (for example `run_job(JOBS[0], "train")`, after rerunning settings).
Increase the cumulative budget to extend a completed run. For a new independent
experiment use a new `RUN_TAG`. Resume retains counters and per-environment reset
ordinals from the last saved snapshot, not an exact snapshot of all asynchronous
simulator/RNG states.

For an old SQLite run, stop all its training processes **before** pulling/restarting.
Resume automatically imports its counters and reset ordinals once, read-only,
into the new JSON format. The old `.sqlite` and any sidecars are retained; do not
delete them before migration. New JSON is preferred on later resumes, preventing
double-counting. An unreadable/corrupt old database fails visibly.

This source change invalidates older smoke certificates. Updated smoke jobs use
the separate `_ram_smoke` suffix and have their own `SMOKE_RUN_MODE`, so the old
full experiment name can be retained. Run fresh RAM smoke tests first. Then set
`RUN_MODE="resume"` for the existing full run and launch only that job, e.g.
`run_job(JOBS[0], "train")`. Do not resume the entire sweep if other jobs have not
been created; those jobs still need fresh mode.

Outputs under `outputs/explore_go` include per-run checkpoints, controller logs,
JSON counter/outcome snapshots, `training_metrics.jsonl` (entropy/value loss/Go returns),
session timings, ordinary-reset episode CSVs and summaries, comparison CSVs and
charts. Final rankings require all planned seeds and use ordinary-reset success,
then original return; they do not select the luckiest training seed. Training Go
returns across different starting-state distributions are diagnostic only.
