"""Notebook subprocess control and evidence-based smoke gate."""
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def fingerprint():
    folder = Path(__file__).resolve().parent
    files = ["explore_go_core.py", "explore_go_sf.py", "explore_go_worker.py",
             "explore_go_control.py", "test_explore_go.py", "home_env.py", "home_models.py", "home_worker.py"]
    return {p:hashlib.sha256((folder/p).read_bytes()).hexdigest() for p in files}


def protocol(job):
    return {key:job[key] for key in ["configuration", "stack", "frame_skip", "learning_rate",
                                    "exploration", "budget_unit", "device"]}


def run_job(job, stage):
    folder = Path(__file__).resolve().parent
    job = dict(job, stage=stage)
    destination = Path(job["root"])/"controller"/job["name"]
    destination.mkdir(parents=True, exist_ok=True)
    token = f"{stage}_{time.time_ns()}"
    job_path = destination/f"{token}.json"
    log_path = destination/f"{token}.log"
    job_path.write_text(json.dumps(job, indent=2))
    print(f"Running {job['name']} {stage}; log: {log_path}", flush=True)
    options = {"start_new_session": True} if os.name == "posix" else {}
    process = subprocess.Popen([sys.executable, "-u", str(folder/"explore_go_worker.py"), str(job_path)],
                               cwd=folder, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, **options)
    try:
        with log_path.open("w") as log:
            for line in process.stdout:
                log.write(line); log.flush(); print(line, end="", flush=True)
        status = process.wait()
        if status: raise RuntimeError(f"{stage} failed (exit {status}); inspect {log_path}")
    except BaseException:
        if process.poll() is None:
            if os.name == "posix": os.killpg(process.pid, signal.SIGINT)
            else: process.terminate()
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                if os.name == "posix": os.killpg(process.pid, signal.SIGKILL)
                else: process.kill()
                process.wait()
        raise
    finally: process.stdout.close()
    return log_path


def accept_smokes(jobs, certificate):
    """Require successful native training, checkpoints and ordinary-reset evaluation for all arms."""
    if {j["configuration"] for j in jobs} != {"baseline", "random_fixed", "random_variable"}:
        raise RuntimeError("Validate all three configurations before full training")
    evidence = []
    for job in jobs:
        if not job["is_smoke"]: raise ValueError("Expected smoke jobs")
        folder = Path(job["root"])/"results"/job["name"]
        train = json.loads((folder/"training_summary.json").read_text())
        evaluation = json.loads((folder/f"evaluation_{job['checkpoint_kind']}"/"summary.json").read_text())
        counts = train["actual"]
        assert train["status"] == 0 and counts["total_"+job["budget_unit"]] >= job["budget"]
        assert counts["total_decisions"] == counts.get("explore_decisions",0)+counts.get("go_decisions",0)
        assert counts["total_game_tics"] == counts.get("explore_game_tics",0)+counts.get("go_game_tics",0)
        assert counts.get("go_decisions",0) > 0
        assert counts.get("valid_go_starts",0) > 0
        if job["configuration"] == "baseline": assert counts.get("explore_decisions",0) == 0
        else: assert counts.get("explore_decisions",0) > 0
        assert evaluation["evaluation_exploration_enabled"] is False
        assert len(evaluation["episodes"]) == len(job["eval_seeds"]) > 0
        assert Path(evaluation["checkpoint"]).is_file()
        assert all(math.isfinite(r["original_return"]) and r["decisions"] > 0 for r in evaluation["episodes"])
        metrics = folder/"training_metrics.jsonl"
        rows = [json.loads(line) for line in metrics.read_text().splitlines()]
        assert any("entropy" in r and "value_loss" in r and math.isfinite(r["value_loss"]) for r in rows)
        evidence.append(dict(name=job["name"], training_summary=str(folder/"training_summary.json"),
                             evaluation=str(folder/f"evaluation_{job['checkpoint_kind']}"/"summary.json"),
                             protocol=protocol(job)))
    Path(certificate).write_text(json.dumps(dict(source_fingerprint=fingerprint(), evidence=evidence), indent=2))
    print("Native training/evaluation smoke gate passed:", certificate)


def check_certificate(path, job=None):
    certificate = json.loads(Path(path).read_text())
    if certificate["source_fingerprint"] != fingerprint():
        raise RuntimeError("Source changed since validation; rerun smoke tests")
    if len(certificate["evidence"]) != 3: raise RuntimeError("Missing three-arm smoke evidence")
    if job is not None and protocol(job) not in [row["protocol"] for row in certificate["evidence"]]:
        raise RuntimeError("Protocol changed since smoke validation; validate the new settings first")
    for row in certificate["evidence"]:
        if json.loads(Path(row["training_summary"]).read_text())["status"] != 0:
            raise RuntimeError("Smoke training failed")
        if json.loads(Path(row["evaluation"]).read_text())["evaluation_exploration_enabled"]:
            raise RuntimeError("Smoke evaluation was not ordinary-reset")
