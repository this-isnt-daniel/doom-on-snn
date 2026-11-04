"""Isolated train/evaluate/replay entry point for the my-way-home notebooks."""
import os
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"]="1"  # trusted checkpoints only
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
import numpy as np
import torch


def select_checkpoint(run_dir,kind):
    paths=list((run_dir/"checkpoint_p0").glob("best_*.pth" if kind=="best" else "checkpoint_*.pth"))
    if not paths:raise FileNotFoundError(f"No {kind} checkpoint under {run_dir}")
    if kind=="latest":return sorted(paths)[-1]
    def rank(p):
        match=re.search(r"_reward_([-+0-9.eE]+)\.pth$",p.name)
        if not match:raise ValueError(f"Unknown best filename: {p.name}")
        return float(match.group(1)),p.name
    return max(paths,key=rank)


def write_csv(path,rows):
    if not rows:return
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main(job):
    import home_models as models
    import home_env
    from sample_factory.cfg.arguments import parse_sf_args,parse_full_cfg,load_from_checkpoint
    from sample_factory.envs.create_env import create_env
    from sample_factory.model.actor_critic import create_actor_critic
    from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
    from sample_factory.train import run_rl
    from sf_examples.vizdoom.doom.doom_params import add_doom_env_args,doom_override_defaults
    from importlib.metadata import version
    home_env.register();models.register_models(job["model"])
    torch.set_num_threads(1)
    if not torch.cuda.is_available():raise RuntimeError("GPU allocation required")
    device=torch.device("cuda")
    torch.ones(1,device=device).item()
    seed=job["seed"]
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    root=Path(job["output_root"]);run_dir=root/"train_dir"/job["experiment"]
    result_dir=root/"results"/job["experiment"]
    if job["stage"]=="train" and job["mode"]=="fresh" and run_dir.exists():
        raise FileExistsError(f"Fresh training refuses existing {run_dir}; use a new RUN_TAG")
    if (job["stage"]!="train" or job["mode"]=="resume") and not (run_dir/"config.json").is_file():
        raise FileNotFoundError(f"No saved run config: {run_dir}")
    result_dir.mkdir(parents=True,exist_ok=True)
    (result_dir/f"job_{job['stage']}.json").write_text(json.dumps(job,indent=2))
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [Path(__file__),Path(models.__file__),Path(home_env.__file__)]}

    def parse(argv,evaluation=False):
        parser,_=parse_sf_args(argv=argv,evaluation=evaluation)
        add_doom_env_args(parser);doom_override_defaults(parser);models.add_args(parser)
        return parse_full_cfg(parser,argv)

    argv=["--env=doom_my_way_home",f"--experiment={job['experiment']}",
          f"--train_dir={root/'train_dir'}",f"--seed={seed}","--device=gpu",
          "--use_rnn=False","--nonlinearity=relu","--actor_critic_share_weights=True",
          "--normalize_input=False","--obs_scale=1","--obs_subtract_mean=0",
          "--res_h=72","--res_w=128",f"--env_frameskip={job['frame_skip']}",
          "--env_framestack=1",f"--num_workers={job['workers']}","--num_envs_per_worker=4",
          f"--batch_size={job['batch_size']}",f"--learning_rate={job['learning_rate']}",
          "--set_workers_cpu_affinity=False",f"--home_model={job['model']}",
          f"--home_T={job['T']}",f"--home_beta={job['beta']}",
          f"--home_stack={job['stack']}",f"--home_logit_scale={job['logit_scale']}"]
    cfg=parse(argv,evaluation=job["stage"]!="train")
    if job["stage"]!="train" or job["mode"]=="resume":cfg=load_from_checkpoint(cfg)
    if cfg.home_model!=job["model"]:raise ValueError("Saved model type differs from notebook")
    env=create_env(cfg.env,cfg=cfg,env_config=None)
    try:
        obs,scenario=home_env.validate(env,seed)
        obs_space,action_space=env.observation_space,env.action_space
        if cfg.normalize_input or cfg.obs_scale!=1:raise ValueError("Rate coding requires uncentered image probabilities")
        print("Scenario controls and reward:",scenario,flush=True)
        (result_dir/"scenario.json").write_text(json.dumps(scenario,indent=2))
        model=create_actor_critic(cfg,obs_space,action_space)
        model.model_to_device(device);model.eval()

        def logits_for(observation,actor=model):
            batch={k:torch.as_tensor(v).unsqueeze(0) for k,v in observation.items()}
            normalized=prepare_and_normalize_obs(actor,batch)
            core,_=actor.forward_core(actor.forward_head(normalized),None)
            result=actor.forward_tail(core,values_only=False,sample_actions=False)
            return result["action_logits"],normalized

        if job["stage"]=="train" and job["mode"]=="fresh" and job["model"]=="snn":
            # Calibrate on the same initialized weights subsequently used by workers.
            env.action_space.seed(seed);frames=[];episode=0
            while len(frames)<job["calibration_frames"]:
                observation,_=home_env.reset_episode(env,seed+episode);episode+=1
                done=truncated=False;decision=0
                while not(done or truncated) and len(frames)<job["calibration_frames"]:
                    if decision%2==0:frames.append(observation["obs"].copy())
                    observation,_,done,truncated,_=env.step(env.action_space.sample());decision+=1
            # Calibration subset limits GPU memory even with image stacking.
            sample=torch.as_tensor(np.stack(frames[:job["calibration_batch"]]),device=device)
            gains,rates=models.calibrate(model,sample,target=job["target_rate"])
            print("Calibrated hidden/output gains:",gains,"rates:",rates,flush=True)
            torch.save({"frames":torch.tensor(np.stack(frames)),"gains":gains,"rates":rates},result_dir/"calibration.pt")
            initial=result_dir/"initial_state.pt"
            state={k:v.detach().cpu() for k,v in model.state_dict().items()}
            torch.save(state,initial)
            cfg.home_init_state=str(initial)
            cfg.home_gains=",".join(str(g) for g in gains)
            del sample,frames,state

        if job["stage"]=="train":
            # A same-observation test catches uncontrolled stochastic spike generation.
            observation,_=home_env.reset_episode(env,seed)
            with torch.no_grad():
                a,_=logits_for(observation);b,_=logits_for(observation)
                assert torch.equal(a,b),"Policy logits must be repeatable for PPO ratios"
            if job["model"]=="snn":
                print("Output rates before training:",(a/cfg.home_logit_scale).cpu().tolist(),flush=True)
                if not torch.any(a):print("WARNING: output silent on probe; inspect calibration",flush=True)
                assert not list(model.action_parameterization.parameters()),"Unexpected learned post-spike readout"
            cfg.train_for_env_steps=job["total_steps"]
            del logits_for  # release its default-argument reference to probe model
            del model
            torch.cuda.empty_cache()
    finally:env.close()

    if job["stage"]=="train":
        started=time.perf_counter();status=run_rl(cfg);elapsed=time.perf_counter()-started
        if status!=0:raise RuntimeError(f"Sample Factory training failed: {status}")
        timing_path=result_dir/"training_sessions.json"
        sessions=json.loads(timing_path.read_text()) if timing_path.exists() else []
        sessions.append({"mode":job["mode"],"requested_total_steps":job["total_steps"],"seconds":elapsed})
        timing_path.write_text(json.dumps(sessions,indent=2))
        print("Training complete; seconds:",elapsed,flush=True)
        return

    # Load selected learned checkpoint only after building the exact saved architecture.
    kind=job["checkpoint_kind"] if job["stage"]=="evaluate" else "best"
    path=select_checkpoint(run_dir,kind)
    checkpoint=torch.load(path,map_location="cpu",weights_only=False)
    model.load_state_dict(checkpoint["model"],strict=True);model.eval()
    with path.open("rb") as f:checkpoint_hash=hashlib.file_digest(f,"sha256").hexdigest()
    output_dir=result_dir/(f"evaluation_{kind}" if job["stage"]=="evaluate" else "replay_best")
    output_dir.mkdir(parents=True,exist_ok=True)
    env=create_env(cfg.env,cfg=cfg,env_config=None,render_mode="rgb_array" if job["stage"]=="replay" else None)
    records=[];latencies=[];activity=np.zeros(6);activity_n=0;ties=np.zeros(2);silent=np.zeros(2);n_decisions=0
    seeds=job["eval_seeds"] if job["stage"]=="evaluate" else job["video_seeds"]
    if job["stage"]=="replay" and not shutil.which("ffmpeg"):
        env.close();raise RuntimeError("ffmpeg with libx264 must be in PATH for replay")
    try:
        probe,_=home_env.reset_episode(env,seeds[0])
        with torch.no_grad():
            for _ in range(5):logits_for(probe)
        torch.cuda.synchronize()
        for ep_seed in seeds:
            obs,_=home_env.reset_episode(env,ep_seed);game=env.unwrapped.game
            initial_tic=game.get_episode_time();done=truncated=False
            total=0.;decisions=0;success=False;encoder=None;video_log=None;actions=[]
            video_path=output_dir/f"seed{ep_seed}.mp4"
            try:
                while not(done or truncated):
                    if job["stage"]=="replay":
                        frame=env.render()
                        if frame is None:raise RuntimeError("No replay RGB frame")
                        frame=np.asarray(frame)
                        if frame.shape[0]==3 and frame.shape[-1]!=3:frame=frame.transpose(1,2,0)
                        frame=np.ascontiguousarray(frame,dtype=np.uint8)
                        if encoder is None:
                            h,w=frame.shape[:2];video_log=video_path.with_suffix(".ffmpeg.log").open("wb")
                            encoder=subprocess.Popen([shutil.which("ffmpeg"),"-y","-f","rawvideo","-pix_fmt","rgb24",
                                "-s",f"{w}x{h}","-r",str(35/cfg.env_frameskip),"-i","pipe:0","-an",
                                "-vf","pad=ceil(iw/2)*2:ceil(ih/2)*2","-c:v","libx264","-pix_fmt","yuv420p",
                                "-movflags","+faststart",str(video_path)],stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL,stderr=video_log)
                        encoder.stdin.write(frame.tobytes())
                    with torch.no_grad():
                        torch.cuda.synchronize();started=time.perf_counter()
                        logits,normalized=logits_for(obs)
                        action=(int(logits[0,:3].argmax()),int(logits[0,3:].argmax()))
                        torch.cuda.synchronize();latencies.append(time.perf_counter()-started)
                        if job["model"]=="snn":
                            groups=logits.reshape(2,3)
                            ties+=(groups==groups.max(dim=1,keepdim=True).values).sum(1).gt(1).cpu().numpy()
                            silent+=(groups==0).all(1).cpu().numpy()
                            if activity_n<job["activity_frames"]:
                                _,counts=model.encoder.run(normalized["obs"],collect=True)
                                activity+=counts.sum(0).cpu().numpy();activity_n+=1
                    obs,reward,done,truncated,_=env.step(action)
                    # Native my-way-home awards +1 on vest collection, small negative living reward otherwise.
                    success=success or float(reward)>.5
                    total+=float(reward);decisions+=1;n_decisions+=1
                    actions.append({"decision":decisions-1,"turn":action[0],"movement":action[1],"reward":float(reward)})
                elapsed_tics=game.get_episode_time()-initial_tic
                timeout=not success and game.get_episode_time()>=game.get_episode_timeout()
                row={"seed":ep_seed,"return":total,"success":success,"timeout":bool(timeout),
                     "decisions":decisions,"elapsed_game_seconds":elapsed_tics/35.,
                     "time_to_goal_seconds":elapsed_tics/35. if success else None}
                records.append(row)
                print(f"episode={ep_seed} return={total:.4f} success={success} decisions={decisions}",flush=True)
                write_csv(output_dir/"episodes.csv",records)
                if job["stage"]=="replay":write_csv(video_path.with_suffix(".actions.csv"),actions)
            finally:
                try:
                    if encoder is not None:
                        try:encoder.stdin.close()
                        finally:
                            try:encoder.wait(timeout=60)
                            except subprocess.TimeoutExpired:
                                encoder.kill();encoder.wait();raise RuntimeError("ffmpeg timed out")
                        if encoder.returncode:raise RuntimeError("Video encoding failed; inspect ffmpeg log")
                finally:
                    if video_log:video_log.close()
    finally:env.close()
    rewards=np.array([r["return"] for r in records]);times=[r["time_to_goal_seconds"] for r in records if r["success"]]
    neurons=np.array([3*cfg.home_stack*72*128,32*17*31,64*7*14,128*3*6,512,6])
    summary={"model":job["model"],"experiment":job["experiment"],"training_seed":seed,
        "checkpoint_kind":kind,"checkpoint":str(path),"checkpoint_sha256":checkpoint_hash,
        "env_steps":int(checkpoint["env_steps"]),"mean_return":float(rewards.mean()),
        "std_return":float(rewards.std()),"SE":float(rewards.std()/np.sqrt(len(rewards))),
        "goal_success_rate":float(np.mean([r["success"] for r in records])),
        "timeout_rate":float(np.mean([r["timeout"] for r in records])),
        "mean_time_to_goal_successes_only":float(np.mean(times)) if times else None,
        "mean_decisions":float(np.mean([r["decisions"] for r in records])),
        "latency_mean_ms":1000*float(np.mean(latencies)),"episodes":records,
        "success_definition":"observed reward > 0.5 (native +1 vest reward)",
        "output_tie_fraction":(ties/n_decisions).tolist() if job["model"]=="snn" else None,
        "output_silent_fraction":(silent/n_decisions).tolist() if job["model"]=="snn" else None,
        "firing_rates_input_hidden_output":(activity/(activity_n*neurons*cfg.home_T)).tolist() if activity_n else None,
        "activity_observations":activity_n,"scenario":scenario,"source_sha256":source_hashes,
        "saved_config":json.loads((run_dir/"config.json").read_text()),
        "packages":{p:version(p) for p in ["torch","sample-factory","vizdoom","gymnasium","numpy"]}}
    if job["model"]=="snn":summary["packages"]["snntorch"]=version("snntorch")
    (output_dir/"summary.json").write_text(json.dumps(summary,indent=2))
    print("SUMMARY",json.dumps({k:v for k,v in summary.items() if k not in ["episodes","saved_config","scenario"]}),flush=True)


if __name__=="__main__":main(json.loads(Path(sys.argv[1]).read_text()))
