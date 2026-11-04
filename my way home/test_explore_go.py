"""Deterministic mechanism tests; native SF/ViZDoom training is a separate smoke gate."""
import ast
from collections import deque
from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import patch
import sys
import multiprocessing as mp
import sqlite3
import time
import json
import gymnasium as gym
import numpy as np
import torch
from explore_go_core import ExploreGo, ExploreSettings, Ledger, native_tics


def update_shared_ledger(ledger, slot, iterations):
    for _ in range(iterations): ledger.add(go_decisions=1, go_game_tics=4)
    ledger.next_reset(slot)
    ledger.close()  # Child close must not persist or stop the parent's saver.


def source_definitions(path, names, namespace):
    """Execute actual source definitions without importing platform-specific SF services."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in nodes} != set(names): raise AssertionError("Upstream API changed")
    for n in nodes: n.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class TinyDoom(gym.Env):
    """Deterministic clock and rewards, with partial terminal actions like ViZDoom."""
    def __init__(self, timeout=80, goal_steps=None, truncation=False):
        self.observation_space = gym.spaces.Box(0, 255, (3,72,128), np.uint8)
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(3), gym.spaces.Discrete(3)))
        self.game = self
        self.timeout, self.goal_steps, self.truncation = timeout, goal_steps or {}, truncation
        self.resets = 0; self.actions = []
    def set_seed(self, seed): self.current_seed = seed
    def get_episode_time(self): return self.clock
    def get_episode_timeout(self): return self.timeout
    def get_living_reward(self): return -.0001
    def reset(self, *, seed=None, **kwargs):
        self.resets += 1; self.clock = 10; self.steps = 0
        self.set_seed(seed)
        return np.zeros((3,72,128), np.uint8), {}
    def step(self, action):
        self.steps += 1; self.actions.append(action)
        ticks = min(4, self.timeout-self.clock); self.clock += ticks
        goal = self.steps == self.goal_steps.get(self.resets)
        ended = goal or self.clock >= self.timeout
        if goal: self.clock = 0  # native terminal clock can disappear
        return np.full((3,72,128), self.steps, np.uint8), float(goal)-ticks*.0001, ended and not self.truncation, ended and self.truncation, {}


class MechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ns = source_definitions(Path(__file__).with_name("home_env.py"), ["HomeImages"],
                                dict(gym=gym, np=np, deque=deque))
        cls.Images = ns["HomeImages"]

    def build(self, settings=None, native=None, stack=4, ledger=None):
        native = native or TinyDoom()
        env = ExploreGo(self.Images(native, stack), settings or ExploreSettings(), 0, "w0", 4, ledger)
        self.addCleanup(env.close)
        return env, native

    def test_zero_matches_unwrapped_observations_rewards_and_boundaries(self):
        env, game = self.build()
        raw = self.Images(TinyDoom(), 4)
        a, _ = env.reset(seed=42); b, _ = raw.reset(seed=42)
        np.testing.assert_array_equal(a["obs"], b["obs"])
        for _ in range(18):
            a, ar, ad, at, _ = env.step((1,2)); b, br, bd, bt, _ = raw.step((1,2))
            np.testing.assert_array_equal(a["obs"], b["obs"])
            self.assertEqual((ar,ad,at), (br,bd,bt))
        self.assertEqual(env.local_counts.get("explore_decisions", 0), 0)

    def test_handover_preserves_stack_clock_and_no_synthetic_transition(self):
        env, game = self.build(ExploreSettings("fixed", 3))
        obs, info = env.reset(seed=0)
        np.testing.assert_allclose(obs["obs"][:,0,0], np.repeat([0,1,2,3],3)/255)
        self.assertEqual(game.resets, 1); self.assertEqual(game.clock, 22)
        self.assertEqual(info["eg_explore_steps"], 3)
        obs, reward, done, trunc, _ = env.step((0,0))
        self.assertEqual((reward, done, trunc), (-.0004, False, False))
        self.assertEqual(env.local_counts["go_decisions"], 1)
        np.testing.assert_allclose(obs["obs"][:,0,0], np.repeat([1,2,3,4],3)/255)

    def test_mixture_and_duration_cap(self):
        env, _ = self.build(ExploreSettings("fixed", 999, explore_probability=.5))
        lengths = [env.reset()[1]["eg_explore_steps"] for _ in range(60)]
        self.assertEqual(set(lengths), {0,4})  # floor((80-10)/4 * .25)

    def test_uniform_includes_zero_and_is_reproducible(self):
        settings = ExploreSettings("uniform", maximum_steps=3)
        a, ga = self.build(settings); b, gb = self.build(settings)
        la = [a.reset()[1]["eg_explore_steps"] for _ in range(40)]
        lb = [b.reset()[1]["eg_explore_steps"] for _ in range(40)]
        self.assertEqual(la, lb); self.assertEqual(ga.actions, gb.actions)
        self.assertEqual(set(la), {0,1,2,3})

    def test_explore_goal_retries_and_prefix_reward_never_enters_go(self):
        env, game = self.build(ExploreSettings("fixed",3), TinyDoom(goal_steps={1:2}))
        env.reset(); self.assertEqual(game.resets,2)
        self.assertEqual(env.local_counts["explore_goal"],1)
        self.assertEqual(env.local_counts["explore_decisions"],5)
        self.assertEqual(env.step((0,0))[1], -.0004)

    def test_timeout_partial_tic_and_no_bootstrap_after_finite_horizon(self):
        env, _ = self.build(native=TinyDoom(timeout=19, truncation=True))
        env.reset()
        for _ in range(2): self.assertEqual(env.step((0,0))[2:4], (False,False))
        _, reward, done, truncated, info = env.step((0,0))
        self.assertEqual((reward, done, truncated), (-.0001,True,False))
        self.assertTrue(info["eg_original_truncated"])
        self.assertEqual(env.local_counts["go_game_tics"],9)

    def test_artificial_truncation_fails_instead_of_incorrect_bootstrap(self):
        env, _ = self.build(native=TinyDoom(goal_steps={1:1}, truncation=True))
        env.reset()
        with self.assertRaisesRegex(RuntimeError, "artificial truncation"): env.step((0,0))

    def test_retry_bound(self):
        settings = ExploreSettings("fixed",3, maximum_reset_attempts=2)
        env, _ = self.build(settings, TinyDoom(goal_steps={1:1,2:1}))
        with self.assertRaisesRegex(RuntimeError, "Too many episodes"): env.reset()
        self.assertEqual(env.local_counts["reset_retry_exhausted"],1)

    def test_explore_timeout_is_logged_and_retried(self):
        class EarlyTimeout(TinyDoom):
            def reset(self, **kwargs):
                self.timeout = 80
                return super().reset(**kwargs)
            def step(self, action):
                if self.resets == 1: self.timeout = 14
                return super().step(action)
        env, game = self.build(ExploreSettings("fixed",3), EarlyTimeout())
        env.reset()
        self.assertEqual(game.resets,2)
        self.assertEqual(env.local_counts["explore_timeout"],1)
        self.assertEqual(env.local_counts["explore_decisions"],4)

    def test_actual_sf_recorder_receives_go_only(self):
        spec = find_spec("sample_factory")
        path = Path(next(iter(spec.submodule_search_locations)))/"algo/sampling/non_batched_sampling.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        actor = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="ActorState")
        recorder = next(n for n in actor.body if isinstance(n,ast.FunctionDef) and n.name=="record_env_step")
        ns = {}
        exec(compile(ast.Module(body=[recorder],type_ignores=[]),str(path),"exec"),ns)
        state = SimpleNamespace(curr_traj_buffer={k:np.zeros(2) for k in ["rewards","dones","time_outs","policy_id"]},
                                is_active=True, curr_policy_id=0, last_episode_duration=0,
                                env_info=SimpleNamespace(frameskip=4), cfg=SimpleNamespace(summaries_use_frameskip=True))
        env, _ = self.build(ExploreSettings("fixed",3))
        env.reset()
        # Actual SF recorder is called only for explicit Go steps, never reset's prefix.
        for i in range(2):
            _, reward, term, trunc, info = env.step((0,0))
            ns["record_env_step"](state,reward,term,trunc,info,i)
        np.testing.assert_allclose(state.curr_traj_buffer["rewards"],[-.0004,-.0004])
        self.assertEqual(env.local_counts["explore_decisions"],3)
        self.assertEqual(env.local_counts["go_decisions"],2)
        self.assertEqual(state.curr_traj_buffer["dones"].sum(),0)

    def test_durable_accounting_and_resume_reset_ordinals(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"counts.json"
            first = Ledger(path)
            self.assertEqual(first.next_reset("w0"),0)
            first.add(explore_decisions=3, explore_game_tics=12, go_decisions=2, go_game_tics=8)
            first.close()
            second = Ledger(path)
            self.assertEqual(second.next_reset("w0"),1)
            self.assertEqual(second.snapshot()["total_game_tics"],20)
            self.assertEqual(second.snapshot()["total_decisions"],5)
            second.close()

    def test_spawned_workers_share_exact_counters_without_disk_writes(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"counts.json"
            ledger = Ledger(path, slots=["a","b"])
            ctx = mp.get_context("spawn")
            processes = [ctx.Process(target=update_shared_ledger,args=(ledger,slot,1000)) for slot in ["a","b"]]
            for process in processes: process.start()
            for process in processes:
                process.join(timeout=45)
                if process.is_alive(): process.terminate(); process.join(); self.fail("Shared-counter child hung")
                self.assertEqual(process.exitcode,0)
            self.assertEqual(ledger.snapshot()["total_decisions"],2000)
            self.assertEqual(ledger.snapshot()["total_game_tics"],8000)
            self.assertFalse(path.exists(),"Worker actions or closes wrote to disk")
            ledger.close()
            restored = Ledger(path,slots=["a","b"])
            self.assertEqual(restored.snapshot()["total_decisions"],2000)
            self.assertEqual(restored.next_reset("a"),1)
            restored.close()

    def test_periodic_save_atomic_replacement_and_final_flush(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"counts.json"
            ledger = Ledger(path,save_interval=.02)
            ledger.start_saving()
            ledger.add(go_decisions=3,go_game_tics=12)
            deadline = time.monotonic()+5
            while time.monotonic() < deadline:
                state = json.loads(path.read_text())
                if state["counts"]["go_decisions"] == 3: break
                time.sleep(.01)
            else: self.fail("Periodic save did not run")
            ledger.add(explore_decisions=2,explore_game_tics=8)
            ledger.close()
            restored = Ledger(path)
            self.assertEqual(restored.snapshot()["total_decisions"],5)
            self.assertEqual(restored.snapshot()["total_game_tics"],20)
            self.assertFalse(list(Path(tmp).glob("*.tmp")))
            restored.close()

    def test_active_parent_saver_can_be_passed_to_spawned_worker(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"counts.json"
            ledger = Ledger(path, save_interval=3600)
            ledger.start_saving()
            before = path.read_bytes()
            try:
                process = mp.get_context("spawn").Process(target=update_shared_ledger,args=(ledger,"w0",10))
                process.start(); process.join(timeout=45)
                if process.is_alive(): process.terminate(); process.join(); self.fail("Spawn with active saver hung")
                self.assertEqual(process.exitcode,0)
                self.assertEqual(ledger.snapshot()["total_decisions"],10)
                self.assertTrue(ledger._thread.is_alive())
                self.assertEqual(path.read_bytes(),before)
            finally: ledger.close()

    def test_read_only_legacy_migration_preserves_counts_and_reset_ordinals(self):
        with TemporaryDirectory() as tmp:
            legacy = Path(tmp)/"counts.sqlite"
            with sqlite3.connect(legacy) as db:
                db.execute("CREATE TABLE counts (key TEXT PRIMARY KEY,value REAL)")
                db.execute("CREATE TABLE resets (slot TEXT PRIMARY KEY,ordinal INTEGER)")
                db.executemany("INSERT INTO counts VALUES (?,?)",[("go_decisions",7),("go_game_tics",28)])
                db.execute("INSERT INTO resets VALUES ('w0',9)")
            db.close()
            original = legacy.read_bytes()
            path = legacy.with_suffix(".json")
            ledger = Ledger(path)
            self.assertEqual(ledger.snapshot()["total_decisions"],7)
            self.assertEqual(ledger.next_reset("w0"),10)
            ledger.add(explore_decisions=1)
            ledger.close()
            self.assertEqual(legacy.read_bytes(),original)
            # Prefer new JSON on subsequent resumes; don't reimport or double-count.
            restored = Ledger(path)
            self.assertEqual(restored.snapshot()["total_decisions"],8)
            restored.close()

    def test_native_terminal_clock_reset_and_reward_validation(self):
        self.assertEqual(native_tics(100,0,.9997,-.0001,4),3)
        with self.assertRaises(ValueError): native_tics(10,14,7.,-.0001,4)


class UpstreamGAETests(unittest.TestCase):
    def test_budget_monitor_counts_both_phases_and_preserves_stats_handler(self):
        ns = source_definitions(Path(__file__).with_name("explore_go_sf.py"), ["attach_budget_monitor"], {})
        with TemporaryDirectory() as tmp:
            ledger = Ledger(Path(tmp)/"ledger.sqlite")
            ledger.add(explore_decisions=7, go_decisions=2)
            original_handler = lambda *args: None
            runner = SimpleNamespace(_should_end_training=lambda:False,
                                     policy_msg_handlers={"train":[original_handler]}, env_steps={0:8})
            with patch.dict(sys.modules, {"sample_factory.algo.utils.misc":SimpleNamespace(TRAIN_STATS="train")}):
                ns["attach_budget_monitor"](runner,ledger,10,"decisions",tmp)
            self.assertFalse(runner._should_end_training())
            ledger.add(go_decisions=1)
            self.assertTrue(runner._should_end_training())
            self.assertIs(runner.policy_msg_handlers["train"][0],original_handler)
            runner.policy_msg_handlers["train"][1](runner,{"train":{"entropy":1.,"value_loss":.2}},0)
            import json
            row = json.loads((Path(tmp)/"training_metrics.jsonl").read_text())
            self.assertEqual(row["total_decisions"],10)
            self.assertEqual(row["value_loss"],.2)
            ledger.close()

    def test_actual_sample_factory_gae_bootstraps_rollout_but_cuts_episode(self):
        spec = find_spec("sample_factory")
        self.assertIsNotNone(spec, "Install Sample Factory to validate upstream GAE")
        path = Path(next(iter(spec.submodule_search_locations)))/"algo/utils/rl_utils.py"
        ns = source_definitions(path, ["calculate_discounted_sum_torch", "gae_advantages"],
                                dict(torch=torch, Tensor=torch.Tensor, Optional=Optional))
        gae = ns["gae_advantages"]
        rewards = torch.tensor([[1.,2.]])
        values = torch.tensor([[.5,.6,10.]])
        valid = torch.ones_like(values)
        # Rollout boundary: bootstrap from V(s_T), with no synthetic done.
        a = gae(rewards, torch.tensor([[False,False]]), values, valid, .9, .95)
        np.testing.assert_allclose(a.numpy(), [[.5+.54+.855*(2-.6+9),2-.6+9]], rtol=1e-6)
        # True terminal: no value from next episode, no advantage leakage.
        a = gae(rewards, torch.tensor([[True,False]]), values, valid, .9, .95)
        self.assertAlmostEqual(a[0,0].item(),.5)
        a = gae(rewards, torch.tensor([[False,True]]), values, valid, .9, .95)
        self.assertAlmostEqual(a[0,1].item(),1.4, places=5)


if __name__ == "__main__": unittest.main(verbosity=2)
