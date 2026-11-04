"""Training-only random-prefix exploration, independent of PPO implementation."""
from dataclasses import dataclass
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import multiprocessing as mp
import os
import threading
import time
import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class ExploreSettings:
    duration: str = "zero"  # zero, fixed, uniform (inclusive endpoints)
    fixed_steps: int = 50
    maximum_steps: int = 100
    explore_probability: float = 1.0
    maximum_episode_fraction: float = .25
    maximum_reset_attempts: int = 16

    def validate(self):
        if self.duration not in {"zero", "fixed", "uniform"}: raise ValueError(self.duration)
        if self.fixed_steps < 0 or self.maximum_steps < 0: raise ValueError("Negative K")
        if not 0 <= self.explore_probability <= 1: raise ValueError("Invalid mixture probability")
        if not 0 <= self.maximum_episode_fraction < 1: raise ValueError("Fraction must be in [0,1)")
        if self.maximum_reset_attempts < 1: raise ValueError("Reset attempts must be positive")


class RandomExplorer:
    """Small independent interface; a learned explorer can implement action/reset later."""
    def reset(self, rng): self.rng = rng

    def action(self, observation, action_space):
        if not isinstance(action_space, gym.spaces.Tuple): raise TypeError(action_space)
        return tuple(int(self.rng.integers(s.n)) for s in action_space.spaces)


class Ledger:
    """Shared RAM counters; only the parent periodically saves atomic JSON snapshots.

    Pass this object through SF's spawned environment factory, never through its
    JSON configuration. Per-action updates take a shared-memory lock, without IPC
    servers or filesystem writes. Forced exits can lose counts since the last save.
    """
    KEYS = (
        "explore_decisions", "explore_game_tics", "explore_reward",
        "go_decisions", "go_game_tics", "go_reward", "native_resets",
        "explore_selected", "ordinary_starts", "exploration_requested_steps",
        "exploration_capped", "explore_goal", "explore_timeout", "explore_other_terminal",
        "exploration_aborted_episodes", "valid_go_starts", "explored_go_starts",
        "reset_retry_exhausted", "go_episodes", "go_goals", "go_timeouts",
        "go_completed_return", "go_completed_decisions",
    )

    def __init__(self, path, slots=None, save_interval=30.):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.slots = tuple(slots if slots is not None else ["w0"])
        if not self.slots or len(set(self.slots)) != len(self.slots): raise ValueError("Invalid ledger slots")
        if save_interval <= 0: raise ValueError("save_interval must be positive")
        self.save_interval = float(save_interval)
        self.owner_pid = os.getpid()
        self._thread = self._stop = None
        self._flush_error = None
        ctx = mp.get_context("spawn")
        self.lock = ctx.RLock()
        self.counts = ctx.Array("d", len(self.KEYS), lock=self.lock)
        self.resets = ctx.Array("q", [-1]*len(self.slots), lock=self.lock)
        self._index = {key:i for i,key in enumerate(self.KEYS)}
        self._slot_index = {slot:i for i,slot in enumerate(self.slots)}
        state = None
        if self.path.exists():
            state = json.loads(self.path.read_text())
            if state.get("format_version") != 1: raise ValueError("Unsupported counter snapshot")
        else:
            # One-time read-only import. Old jobs MUST be stopped before resuming.
            legacy = self.path.with_suffix(".sqlite")
            if legacy.exists():
                with closing(sqlite3.connect(legacy.resolve().as_uri()+"?mode=ro", uri=True)) as db:
                    state = dict(counts=dict(db.execute("SELECT key,value FROM counts")),
                                 resets=dict(db.execute("SELECT slot,ordinal FROM resets")))
                print(f"Imported stopped-run SQLite counters: {legacy}; original retained", flush=True)
        if state:
            unknown = set(state["resets"])-set(self.slots)
            if unknown: raise ValueError(f"Saved environment slots differ: {unknown}")
            for key, value in state["counts"].items(): self.counts[self._index[key]] = value
            for slot, ordinal in state["resets"].items(): self.resets[self._slot_index[slot]] = ordinal

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_thread"] = state["_stop"] = None
        return state

    def add(self, **increments):
        with self.lock:
            for key, value in increments.items():
                self.counts[self._index[key]] += float(value)

    def next_reset(self, slot):
        with self.lock:
            index = self._slot_index[slot]
            self.resets[index] += 1
            return int(self.resets[index])

    def event(self, slot, kind, **details):
        # Aggregate outcome counts are already updated by ExploreGo. Intentionally
        # omit per-reset event files to keep worker interaction paths disk-free.
        pass

    def snapshot(self):
        with self.lock: result = dict(zip(self.KEYS, self.counts[:]))
        for unit in ["decisions", "game_tics"]:
            result[f"total_{unit}"] = result.get(f"explore_{unit}", 0) + result.get(f"go_{unit}", 0)
        return result

    def flush(self):
        if os.getpid() != self.owner_pid: return
        # Release the simulator lock BEFORE any serialization or filesystem IO.
        with self.lock:
            state = dict(format_version=1, saved_at=time.time(),
                         counts=dict(zip(self.KEYS,self.counts[:])),
                         resets=dict(zip(self.slots,self.resets[:])))
        temporary = self.path.with_name(self.path.name+f".{self.owner_pid}.tmp")
        with temporary.open("w") as f:
            json.dump(state,f,indent=2); f.flush(); os.fsync(f.fileno())
        os.replace(temporary,self.path)

    def start_saving(self):
        if os.getpid() != self.owner_pid or self._thread is not None: return
        self.flush()
        self._stop = threading.Event()
        def save_loop():
            while not self._stop.wait(self.save_interval):
                try: self.flush()
                except Exception as exc:
                    self._flush_error = exc
                    print(f"Counter snapshot failed: {exc}", flush=True)
        self._thread = threading.Thread(target=save_loop,daemon=True,name="counter-snapshots")
        self._thread.start()

    def close(self):
        if os.getpid() != self.owner_pid: return
        if self._thread is not None:
            self._stop.set(); self._thread.join(); self._thread = None
        self.flush()


def native_tics(before, after, reward, living_reward, frame_skip):
    """Count true tics, including a terminal partial action.

    ViZDoom versions can report episode_time=0 after vest collection. Native
    my-way-home reward (goal=1 plus living_reward*tics) provides an independent
    exact tic count in that case. Reject changed reward schemes rather than guess.
    """
    if living_reward >= 0: raise ValueError("Expected negative native living reward")
    goal = 1. if reward > .5 else 0.
    raw = (float(reward)-goal)/living_reward
    inferred = int(round(raw))
    if inferred < 1 or inferred > frame_skip or abs(raw-inferred) > 1e-3:
        raise ValueError(f"Cannot infer tics from native reward: {reward}; check scenario/reward wrappers")
    delta = after-before
    if 0 < delta <= frame_skip and delta != inferred:
        raise ValueError(f"Game clock/reward tic counts disagree: {delta}, {inferred}")
    return inferred


class ExploreGo(gym.Wrapper):
    """Hide Explore transitions in reset; expose unchanged Go steps to PPO.

    Place OUTSIDE the frame-stack wrapper. This makes exploration update the real
    stack, preserving it at handover. No reset, synthetic reward or done is emitted
    at the Explore-to-Go boundary.
    """
    def __init__(self, env, settings, seed, slot, frame_skip, ledger=None, explorer=None):
        super().__init__(env)
        settings.validate()
        self.settings, self.seed, self.slot = settings, int(seed), str(slot)
        self.frame_skip, self.ledger = int(frame_skip), ledger
        self.explorer = explorer or RandomExplorer()
        self.ordinal = -1
        self.local_counts = {}
        self.ready = False

    def _add(self, **increments):
        for key, value in increments.items(): self.local_counts[key] = self.local_counts.get(key, 0) + value
        if self.ledger: self.ledger.add(**increments)

    def _event(self, kind, **details):
        if self.ledger: self.ledger.event(self.slot, kind, **details)

    def _reset_native(self, seed, kwargs):
        game = self.unwrapped.game
        if game is not None: game.set_seed(seed)
        return self.env.reset(seed=seed, **kwargs)

    def _perform(self, action, phase):
        game = self.unwrapped.game
        before = game.get_episode_time()
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)
        tics = native_tics(before, game.get_episode_time(), reward, game.get_living_reward(), self.frame_skip)
        self.elapsed_tics += tics
        self._add(**{f"{phase}_decisions": 1, f"{phase}_game_tics": tics, f"{phase}_reward": reward})
        goal = reward > .5
        # Only native finite-horizon termination is supported here. It is part of
        # the task, not a continuing-task time-limit requiring V(terminal_obs).
        finite_timeout = not goal and self.start_tic + self.elapsed_tics >= game.get_episode_timeout()
        if truncated and not finite_timeout:
            raise RuntimeError("Unexpected artificial truncation: unsupported terminal-observation bootstrap; disable the external time limit")
        info = dict(info, eg_phase=phase, eg_game_tics=tics, eg_goal=goal,
                    eg_finite_timeout=finite_timeout)
        if finite_timeout and (terminated or truncated):
            # Preserve the observed flag in metadata, but represent the finite
            # task horizon as terminal so GAE does not bootstrap past the task.
            info["eg_original_truncated"] = bool(truncated)
            terminated, truncated = True, False
        return obs, reward, bool(terminated), bool(truncated), info

    def reset(self, *, seed=None, **kwargs):
        self.ready = False
        for attempt in range(self.settings.maximum_reset_attempts):
            if self.ledger: ordinal = self.ledger.next_reset(self.slot)
            else:
                self.ordinal += 1
                ordinal = self.ordinal
            # Separate streams: prefix RNG cannot alter policy initialization or
            # Sample Factory's action-sampling RNG. Slots have deterministic hashes.
            import hashlib
            slot_id = int.from_bytes(hashlib.sha256(self.slot.encode()).digest()[:4], "little")
            root_seed = self.seed if seed is None else int(seed)
            rng = np.random.default_rng(np.random.SeedSequence([root_seed, slot_id, ordinal]))
            env_seed = int(rng.integers(0, 2**32-1)) if seed is None or attempt else int(seed)
            obs, info = self._reset_native(env_seed, kwargs)
            self._add(native_resets=1)
            game = self.unwrapped.game
            self.start_tic = int(game.get_episode_time())
            self.elapsed_tics = 0
            setting = self.settings
            selected = setting.duration != "zero" and rng.random() < setting.explore_probability
            requested = 0
            if selected:
                requested = setting.fixed_steps if setting.duration == "fixed" else int(rng.integers(setting.maximum_steps+1))
            remaining = max(0, game.get_episode_timeout()-self.start_tic)
            remaining_decisions = remaining // self.frame_skip
            cap = min(int(remaining_decisions*setting.maximum_episode_fraction), max(0, remaining_decisions-1))
            length = min(requested, cap)
            self._add(explore_selected=int(selected), ordinary_starts=int(length == 0),
                      exploration_requested_steps=requested, exploration_capped=int(length < requested))
            self.explorer.reset(rng)
            ended = False
            for _ in range(length):
                obs, _, terminated, truncated, transition_info = self._perform(self.explorer.action(obs, self.action_space), "explore")
                if terminated or truncated:
                    ended = True
                    outcome = "goal" if transition_info["eg_goal"] else "timeout" if transition_info["eg_finite_timeout"] else "other_terminal"
                    self._add(**{f"explore_{outcome}": 1}, exploration_aborted_episodes=1)
                    self._event("explore_ended", outcome=outcome, requested=requested, seed=env_seed)
                    break
            if ended: continue
            self._add(valid_go_starts=1, explored_go_starts=int(length > 0))
            self._event("handover", seed=env_seed, requested=requested, actual=length,
                        remaining_tics=game.get_episode_timeout()-self.start_tic-self.elapsed_tics)
            self.ready = True
            self.go_return = 0.
            self.go_steps = 0
            return obs, dict(info, eg_explore_steps=length, eg_requested_steps=requested)
        self._add(reset_retry_exhausted=1)
        raise RuntimeError("Too many episodes ended during exploration; lower K/probability or inspect explorer")

    def step(self, action):
        if not self.ready: raise RuntimeError("reset() must complete before Go step")
        obs, reward, terminated, truncated, info = self._perform(action, "go")
        self.go_return += reward
        self.go_steps += 1
        if terminated or truncated:
            self.ready = False
            self._add(go_episodes=1, go_goals=int(info["eg_goal"]), go_timeouts=int(info["eg_finite_timeout"]))
            self._add(go_completed_return=self.go_return, go_completed_decisions=self.go_steps)
            self._event("go_ended", original_go_return=self.go_return, decisions=self.go_steps,
                        goal=info["eg_goal"], timeout=info["eg_finite_timeout"])
        return obs, reward, terminated, truncated, info

    def close(self):
        # Shared counter lifetime belongs to the parent, not individual envs.
        self.env.close()
