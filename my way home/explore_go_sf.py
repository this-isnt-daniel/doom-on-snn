"""Sample Factory integration; model and PPO/GAE are reused unchanged."""
import functools
from explore_go_core import ExploreGo, ExploreSettings
import home_env


def settings(cfg):
    return ExploreSettings(cfg.eg_duration, cfg.eg_fixed, cfg.eg_maximum,
                           cfg.eg_probability, cfg.eg_fraction, cfg.eg_reset_attempts)


def make_env(spec, name, cfg, env_config, render_mode=None, ledger=None, **kwargs):
    env = home_env.make_home(spec, name, cfg, env_config, render_mode, **kwargs)
    # Environment-info probes and all evaluation/replay use ordinary resets.
    if not cfg.eg_training or env_config is None: return env
    slot = "_".join(f"{k}{env_config.get(k, 0)}" for k in ["worker_index", "vector_index", "env_id"])
    if ledger is None: raise RuntimeError("Training requires the parent's shared counter ledger")
    return ExploreGo(env, settings(cfg), cfg.seed, slot, cfg.env_frameskip, ledger)


def register(ledger=None):
    from sample_factory.envs.env_utils import register_env
    from sf_examples.vizdoom.doom.doom_utils import DOOM_ENVS
    spec = next(s for s in DOOM_ENVS if s.name == "doom_my_way_home")
    register_env(spec.name, functools.partial(make_env, spec, ledger=ledger))


def add_args(parser):
    from sample_factory.utils.utils import str2bool
    parser.add_argument("--eg_training", type=str2bool, default=False)
    parser.add_argument("--eg_duration", choices=["zero", "fixed", "uniform"], default="zero")
    parser.add_argument("--eg_fixed", type=int, default=50)
    parser.add_argument("--eg_maximum", type=int, default=100)
    parser.add_argument("--eg_probability", type=float, default=.5)
    parser.add_argument("--eg_fraction", type=float, default=.25)
    parser.add_argument("--eg_reset_attempts", type=int, default=16)
    parser.add_argument("--eg_ledger", type=str, default="")


def attach_budget_monitor(runner, ledger, budget, unit, destination):
    """Adjust runner stopping, never the learner's rollout/advantages/losses.

    Sampling is asynchronous, so stop occurs at a training-iteration boundary and
    in-flight interactions can overshoot. All such work remains in the ledger.
    """
    import json
    import time
    from pathlib import Path
    from sample_factory.algo.utils.misc import TRAIN_STATS
    if unit not in {"decisions", "game_tics"}: raise ValueError(unit)
    destination = Path(destination)
    original = runner._should_end_training
    last_print = [0.]

    def should_stop():
        if ledger._flush_error is not None:
            raise RuntimeError("Periodic counter snapshot failed; inspect storage") from ledger._flush_error
        counts = ledger.snapshot()
        if time.monotonic()-last_print[0] > 15:
            print("EXPLORE_GO_COUNTS", json.dumps(counts), flush=True)
            last_print[0] = time.monotonic()
        return counts[f"total_{unit}"] >= budget or original()

    runner._should_end_training = should_stop
    handlers = runner.policy_msg_handlers.get(TRAIN_STATS)
    if not isinstance(handlers, list):
        raise RuntimeError("Unsupported Sample Factory training-stats API; validate integration first")

    def stats_callback(r, message, policy_id):
        stats = message[TRAIN_STATS]
        row = {"learner_env_steps": int(r.env_steps.get(policy_id, 0)), **ledger.snapshot()}
        for key in ["entropy", "value_loss", "policy_loss", "exploration_loss", "kl_divergence"]:
            if key in stats:
                value = stats[key]
                row[key] = float(value.item() if hasattr(value, "item") else value)
        with (destination/"training_metrics.jsonl").open("a") as f: f.write(json.dumps(row)+"\n")

    handlers.append(stats_callback)
