"""One isolated training candidate; invoked by the sweep notebook, not on import."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"  # trusted project checkpoints only
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch


def main(job):
    import sweep_support as support
    from sample_factory.cfg.arguments import parse_sf_args, parse_full_cfg, load_from_checkpoint
    from sample_factory.envs.create_env import create_env
    from sample_factory.model.actor_critic import create_actor_critic
    from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
    from sample_factory.train import run_rl
    from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
    from vizdoom import GameVariable
    from importlib.metadata import version
    support.register()
    seed = job["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)
    if not torch.cuda.is_available(): raise RuntimeError("GPU allocation required")
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    # Fail before expensive calibration if CUDA cannot initialize.
    torch.ones(1, device=device).item()
    root = Path(job["root"])
    run_dir = root/"train_dir"/job["name"]
    if run_dir.exists():
        raise FileExistsError(f"Fresh candidate refuses existing run: {run_dir}. Use a new sweep tag.")
    results = root/"results"/job["name"]
    results.mkdir(parents=True, exist_ok=True)
    (results/"job.json").write_text(json.dumps(job, indent=2))

    def parse(argv, evaluation=False):
        parser, _ = parse_sf_args(argv=argv, evaluation=evaluation)
        add_doom_env_args(parser); doom_override_defaults(parser); support.add_args(parser)
        return parse_full_cfg(parser, argv)

    c = job["config"]
    argv = ["--env=doom_defend_the_center", f"--experiment={job['name']}",
            f"--train_dir={root/'train_dir'}", f"--seed={seed}", "--device=gpu",
            "--use_rnn=False", "--nonlinearity=relu", "--actor_critic_share_weights=True",
            "--res_h=72", "--res_w=128", "--env_framestack=1", "--normalize_input_keys=obs",
            "--obs_scale=1.0", "--obs_subtract_mean=0.0",
            "--set_workers_cpu_affinity=False", f"--num_workers={job['workers']}",
            "--num_envs_per_worker=4", "--batch_size=1024",
            f"--learning_rate={c['learning_rate']}", f"--env_frameskip={c['frame_skip']}",
            f"--sweep_stack={c['stack']}", f"--sweep_ammo={c['ammo']}",
            f"--sweep_penalty={c['penalty']}", "--snn_T=8", "--snn_beta=0.9"]
    probe = parse(argv+["--sweep_eval=True"], evaluation=True)
    env = create_env(probe.env, cfg=probe, env_config=None)
    assert [s.n for s in env.action_space.spaces] == [3, 2]
    obs_space, action_space = env.observation_space, env.action_space
    frames = []
    env.action_space.seed(seed)
    try:
        episode = 0
        while len(frames) < job["calibration_frames"]:
            obs, _ = env.reset(seed=seed+episode); episode += 1
            done = truncated = False; decision = 0
            while not (done or truncated) and len(frames) < job["calibration_frames"]:
                if decision % 2 == 0: frames.append(obs["obs"].copy())
                obs, _, done, truncated, _ = env.step(env.action_space.sample()); decision += 1
    finally: env.close()
    x = torch.tensor(np.stack(frames)).float()  # wrapper already scaled pixels
    x = ((x-x.mean(0, keepdim=True))/torch.sqrt(x.var(0, unbiased=False, keepdim=True)+1e-5)).clamp(-5,5)
    gains, rates = support.calibrate_gains(x, T=8, target_rate=.15, device=device, seed=seed, beta=.9)
    torch.save({"frames":x, "gains":gains, "rates":rates}, results/"calibration.pt")
    del x, frames
    argv += ["--snn_gains="+",".join(f"{g:.6f}" for g in gains)]
    cumulative_training = 0.

    for target in job["milestones"]:
        cfg = parse(argv)
        cfg.train_for_env_steps = target  # total target, explicitly override saved value
        start = time.perf_counter()
        status = run_rl(cfg)
        cumulative_training += time.perf_counter()-start
        if status != 0: raise RuntimeError(f"Training failed: {status}")
        checkpoints = sorted((run_dir/"checkpoint_p0").glob("checkpoint_*.pth"))
        if not checkpoints: raise FileNotFoundError("Training produced no checkpoint")
        path = checkpoints[-1]
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        ecfg = load_from_checkpoint(parse(argv, evaluation=True))
        ecfg.sweep_eval = True  # unshaped evaluation even for ammo-penalty candidates
        model = create_actor_critic(ecfg, obs_space, action_space)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.model_to_device(device); model.eval()
        env = create_env(ecfg.env, cfg=ecfg, env_config=None)
        records = []; latencies = []; spike_sum = np.zeros(4); n_activity = 0
        neurons = np.array([32*17*31,64*7*14,128*3*6,512])

        def stats():
            game = env.unwrapped.game
            return [float(game.get_game_variable(v)) for v in
                    [GameVariable.AMMO2, GameVariable.KILLCOUNT]]

        def logits_for(obs):
            batch = {k:torch.as_tensor(v).unsqueeze(0) for k,v in obs.items()}
            normalized = prepare_and_normalize_obs(model, batch)
            core, _ = model.forward_core(model.forward_head(normalized), None)
            logits = model.forward_tail(core, values_only=False, sample_actions=False)["action_logits"]
            return logits, normalized

        try:
            obs, _ = env.reset(seed=job["eval_seeds"][0])
            with torch.no_grad():
                for _ in range(10): logits_for(obs)
            torch.cuda.synchronize()
            for ep_seed in job["eval_seeds"]:
                obs, _ = env.reset(seed=ep_seed)
                start_tic = env.unwrapped.game.get_episode_time()
                done = truncated = False; total=used=kills=0.; decisions=0
                while not (done or truncated):
                    before = stats()
                    with torch.no_grad():
                        torch.cuda.synchronize(); t0=time.perf_counter()
                        logits, normalized = logits_for(obs)
                        action = (int(logits[0,:3].argmax()), int(logits[0,3:].argmax()))
                        torch.cuda.synchronize(); latencies.append(time.perf_counter()-t0)
                        # Separate diagnostic pass, excluded from inference timing.
                        if n_activity < job["activity_frames"]:
                            _, counts = model.encoder.trunk(normalized["obs"], return_spikes=True)
                            spike_sum += counts.sum(0).cpu().numpy(); n_activity += 1
                    obs, reward, done, truncated, info = env.step(action)
                    after = stats()
                    assert abs(float(reward)-info["original_reward"]) < 1e-6
                    total+=info["original_reward"]; used+=max(0.,before[0]-after[0]); kills+=max(0.,after[1]-before[1]); decisions+=1
                elapsed_tics = env.unwrapped.game.get_episode_time()-start_tic
                row = {"seed":ep_seed,"return":total,"kills":kills,"rounds":used,
                       "decisions":decisions,"game_seconds":elapsed_tics/35.}
                records.append(row)
                print(f"{job['name']} target={target} episode={ep_seed} return={total} kills/round={kills/used if used else None}",flush=True)
        finally: env.close()
        rewards=np.array([r["return"] for r in records]); rounds_total=sum(r["rounds"] for r in records)
        firing=spike_sum/(n_activity*neurons*ecfg.snn_T) if n_activity else None
        with path.open("rb") as f: digest=hashlib.file_digest(f,"sha256").hexdigest()
        metrics={"name":job["name"],"category":job["category"],"seed":seed,"config":c,
            "target_steps":target,"actual_steps":int(checkpoint["env_steps"]),
            "checkpoint":str(path),"checkpoint_sha256":digest,"selection":"latest at milestone",
            "mean_return":float(rewards.mean()),"std_return":float(rewards.std()),
            "mean_kills":float(np.mean([r["kills"] for r in records])),
            "kills_per_round":sum(r["kills"] for r in records)/rounds_total if rounds_total else None,
            "mean_decisions":float(np.mean([r["decisions"] for r in records])),
            "mean_game_seconds":float(np.mean([r["game_seconds"] for r in records])),
            "training_seconds":cumulative_training,"latency_mean_ms":1000*float(np.mean(latencies)),
            "latency_p95_ms":1000*float(np.percentile(latencies,95)),
            "activity_observations":n_activity,"firing_rates":firing.tolist() if firing is not None else None,
            "zero_event_sparsity":(1-firing).tolist() if firing is not None else None,
            "episodes":records,"saved_config":json.loads((run_dir/"config.json").read_text()),
            "packages":{p:version(p) for p in ["torch","sample-factory","vizdoom","gymnasium","numpy"]}}
        (results/f"metrics_{target}.json").write_text(json.dumps(metrics,indent=2))
        with (results/f"episodes_{target}.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        del model, checkpoint
        torch.cuda.empty_cache()
    (results/"complete.json").write_text(json.dumps(job,indent=2))


if __name__ == "__main__": main(json.loads(Path(sys.argv[1]).read_text()))
