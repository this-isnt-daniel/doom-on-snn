"""ANN-only Explore-Go-inspired experiment runner (random prefixes, not learned exploration)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch


def run(job):
    import home_models
    import home_env
    import explore_go_sf as integration
    from explore_go_core import Ledger
    from sample_factory.cfg.arguments import parse_sf_args, parse_full_cfg, load_from_checkpoint
    from sample_factory.envs.create_env import create_env
    from sample_factory.model.actor_critic import create_actor_critic
    from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
    from sample_factory.train import make_runner
    from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
    from importlib.metadata import version
    from home_worker import select_checkpoint, write_csv
    if job["stage"] == "train":
        if not job["is_smoke"]:
            from explore_go_control import check_certificate
            check_certificate(job["preflight_certificate"], job)
        import unittest
        import test_explore_go
        suite = unittest.defaultTestLoader.loadTestsFromModule(test_explore_go)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful(): raise RuntimeError("Explore-Go smoke/unit tests failed; training blocked")
    home_models.register_models("ann"); integration.register()
    seed = job["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(1)
    device = torch.device(job.get("device", "cuda"))
    if device.type == "cuda": torch.ones(1, device=device).item(); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    root = Path(job["root"]); name = job["name"]
    run_dir = root/"train_dir"/name
    out = root/"results"/name
    out.mkdir(parents=True, exist_ok=True)
    ledger_path = root/"ledgers"/f"{name}.json"
    legacy_ledger_path = ledger_path.with_suffix(".sqlite")
    training = job["stage"] == "train"
    if training and job["mode"] == "fresh" and (run_dir.exists() or ledger_path.exists() or legacy_ledger_path.exists()):
        raise FileExistsError("Fresh run refuses an existing model or ledger; select a new experiment tag")
    if not training or job["mode"] == "resume":
        if not (run_dir/"config.json").is_file(): raise FileNotFoundError(run_dir/"config.json")
        if training and not (ledger_path.exists() or legacy_ledger_path.exists()):
            raise FileNotFoundError("Cannot resume without counter snapshot or legacy ledger")
    architecture = dict(seed=seed, stack=job["stack"], frame_skip=job["frame_skip"],
                        learning_rate=job["learning_rate"], batch_size=job["batch_size"],
                        workers=job["workers"], envs_per_worker=job["envs_per_worker"],
                        exploration=job["exploration"], budget_unit=job["budget_unit"])
    manifest_path = out/"manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != architecture:
            raise RuntimeError("Architecture/protocol changed on existing run; use a new name")
    else: manifest_path.write_text(json.dumps(architecture, indent=2))
    (out/f"job_{job['stage']}.json").write_text(json.dumps(job, indent=2))

    def parse(argv, evaluation=False):
        parser, _ = parse_sf_args(argv=argv, evaluation=evaluation)
        add_doom_env_args(parser); doom_override_defaults(parser)
        home_models.add_args(parser); integration.add_args(parser)
        return parse_full_cfg(parser, argv)

    e = job["exploration"]
    argv = ["--env=doom_my_way_home", f"--experiment={name}", f"--train_dir={root/'train_dir'}",
            f"--seed={seed}", f"--device={device.type if device.type=='cpu' else 'gpu'}",
            "--home_model=ann", f"--home_stack={job['stack']}", "--use_rnn=False",
            "--actor_critic_share_weights=True", "--nonlinearity=relu",
            "--res_h=72", "--res_w=128", f"--env_frameskip={job['frame_skip']}", "--env_framestack=1",
            "--normalize_input=False", "--obs_scale=1", "--obs_subtract_mean=0",
            f"--num_workers={job['workers']}", f"--num_envs_per_worker={job['envs_per_worker']}",
            f"--batch_size={job['batch_size']}", f"--learning_rate={job['learning_rate']}",
            "--set_workers_cpu_affinity=False", "--with_vtrace=False", "--value_bootstrap=False",
            "--decorrelate_envs_on_one_worker=False", "--decorrelate_experience_max_seconds=0",
            "--experiment_summaries_interval=1",
            f"--eg_duration={e['duration']}", f"--eg_fixed={e['fixed_steps']}",
            f"--eg_maximum={e['maximum_steps']}", f"--eg_probability={e['explore_probability']}",
            f"--eg_fraction={e['maximum_episode_fraction']}", f"--eg_ledger={ledger_path}"]
    cfg = parse(argv, evaluation=not training)
    if not training or job["mode"] == "resume": cfg = load_from_checkpoint(cfg)
    if cfg.home_model != "ann" or cfg.use_rnn or cfg.with_vtrace or cfg.value_bootstrap:
        raise RuntimeError("Unexpected model/GAE protocol in saved configuration")
    # Environment-info probe is deliberately ordinary reset, without any interactions.
    cfg.eg_training = False
    probe = create_env(cfg.env, cfg=cfg, env_config=None)
    try:
        _, scenario = home_env.validate(probe, seed)
        obs_space, action_space = probe.observation_space, probe.action_space
    finally: probe.close()
    (out/"scenario.json").write_text(json.dumps(scenario, indent=2))

    if training:
        cfg.eg_training = True
        cfg.eg_ledger = str(ledger_path)
        # Disable the Go-only stopping budget; monitor the ledger's total instead.
        cfg.train_for_env_steps = 10**15
        slots = [f"worker_index{w}_vector_index{v}_env_id{w*job['envs_per_worker']+v}"
                 for w in range(job["workers"]) for v in range(job["envs_per_worker"])]
        ledger = Ledger(ledger_path, slots, job.get("counter_save_seconds",30.))
        integration.register(ledger)
        before = ledger.snapshot()
        if before[f"total_{job['budget_unit']}"] >= job["budget"]:
            ledger.close(); raise RuntimeError("Total budget already spent; evaluate or increase the explicit total target")
        started = time.perf_counter()
        cfg, runner = make_runner(cfg)
        # make_runner can reload config on resume; enforce runtime-only flags again.
        runner.cfg.eg_training = True
        runner.cfg.eg_ledger = str(ledger_path)
        runner.cfg.train_for_env_steps = 10**15
        integration.attach_budget_monitor(runner, ledger, job["budget"], job["budget_unit"], out)
        try:
            ledger.start_saving()
            status = runner.init()
            if status == 0: status = runner.run()
        finally:
            ledger.close()
        after = ledger.snapshot()
        summary = dict(requested_budget=job["budget"], budget_unit=job["budget_unit"],
                       actual=after, before=before, seconds=time.perf_counter()-started,
                       status=int(status), overshoot=after[f"total_{job['budget_unit']}"]-job["budget"],
                       learner_env_steps=runner.env_steps)
        history_path = out/"training_sessions.json"
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        history.append(summary); history_path.write_text(json.dumps(history, indent=2))
        (out/"training_summary.json").write_text(json.dumps(summary, indent=2))
        if status != 0: raise RuntimeError(f"Sample Factory failed: {status}")
        print("TRAINING_COMPLETE", json.dumps(summary), flush=True)
        return

    # Evaluation always uses ordinary native resets. Never construct ExploreGo.
    cfg.eg_training = False
    checkpoint_path = select_checkpoint(run_dir, job["checkpoint_kind"])
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = create_actor_critic(cfg, obs_space, action_space)
    model.load_state_dict(ckpt["model"], strict=True); model.model_to_device(device); model.eval()
    env = create_env(cfg.env, cfg=cfg, env_config=None)
    rows = []
    try:
        for ep_seed in job["eval_seeds"]:
            obs, _ = home_env.reset_episode(env, ep_seed)
            initial_tic = env.unwrapped.game.get_episode_time()
            done = truncated = False; total = 0.; decisions = 0; tics = 0; goal = False
            from explore_go_core import native_tics
            while not (done or truncated):
                with torch.no_grad():
                    batch = {k:torch.as_tensor(v).unsqueeze(0) for k,v in obs.items()}
                    normalized = prepare_and_normalize_obs(model, batch)
                    core, _ = model.forward_core(model.forward_head(normalized), None)
                    logits = model.forward_tail(core, values_only=False, sample_actions=False)["action_logits"]
                    action = (int(logits[0,:3].argmax()), int(logits[0,3:].argmax()))
                game = env.unwrapped.game
                before = game.get_episode_time()
                obs, reward, done, truncated, _ = env.step(action)
                tics += native_tics(before, game.get_episode_time(), reward, game.get_living_reward(), cfg.env_frameskip)
                total += float(reward); decisions += 1; goal = goal or reward > .5
            rows.append(dict(seed=ep_seed, original_return=total, success=bool(goal),
                             decisions=decisions, game_tics=tics, game_seconds=tics/35.,
                             timeout=bool(not goal and tics + initial_tic >= game.get_episode_timeout())))
            print("EVALUATION", name, rows[-1], flush=True)
    finally: env.close()
    evaluation_dir = out/f"evaluation_{job['checkpoint_kind']}"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    write_csv(evaluation_dir/"episodes.csv", rows)
    with checkpoint_path.open("rb") as f: digest = hashlib.file_digest(f, "sha256").hexdigest()
    reward_array = np.array([r["original_return"] for r in rows])
    stats = dict(name=name, seed=seed, configuration=job["configuration"], checkpoint=str(checkpoint_path),
                 checkpoint_sha256=digest, goal_success_rate=float(np.mean([r["success"] for r in rows])),
                 mean_return=float(reward_array.mean()), std_return=float(reward_array.std()),
                 mean_episode_decisions=float(np.mean([r["decisions"] for r in rows])),
                 mean_episode_game_seconds=float(np.mean([r["game_seconds"] for r in rows])),
                 episodes=rows, evaluation_exploration_enabled=False,
                 actual_training=json.loads((out/"training_summary.json").read_text()),
                 total_training_seconds=sum(s["seconds"] for s in json.loads((out/"training_sessions.json").read_text())),
                 packages={p:version(p) for p in ["torch", "sample-factory", "vizdoom", "gymnasium", "numpy"]})
    (evaluation_dir/"summary.json").write_text(json.dumps(stats, indent=2))
    print("EVALUATION_COMPLETE", json.dumps({k:v for k,v in stats.items() if k!="episodes"}), flush=True)


if __name__ == "__main__": run(json.loads(Path(sys.argv[1]).read_text()))
