"""Training-only random-prefix exploration, independent of PPO implementation."""
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
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
    """Durable, transaction-safe counts shared across sampling processes.

    Counts are committed after each real env.step. Restart retains all observed
    interactions, even if they weren't learned from before interruption. A hard
    process/node crash between a simulator action and its commit cannot be atomic.
    """
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS counts (key TEXT PRIMARY KEY,value REAL NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS resets (slot TEXT PRIMARY KEY,ordinal INTEGER NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY,slot TEXT,kind TEXT,details TEXT)")
        self.db.commit()

    def add(self, **increments):
        with self.db:
            for key, value in increments.items():
                self.db.execute("INSERT INTO counts VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=value+excluded.value",
                                (key, float(value)))

    def next_reset(self, slot):
        with self.db:
            self.db.execute("INSERT INTO resets VALUES (?,0) ON CONFLICT(slot) DO UPDATE SET ordinal=ordinal+1", (slot,))
            return int(self.db.execute("SELECT ordinal FROM resets WHERE slot=?", (slot,)).fetchone()[0])

    def event(self, slot, kind, **details):
        with self.db:
            self.db.execute("INSERT INTO events(slot,kind,details) VALUES (?,?,?)", (slot, kind, json.dumps(details)))

    def snapshot(self):
        result = dict(self.db.execute("SELECT key,value FROM counts"))
        for unit in ["decisions", "game_tics"]:
            result[f"total_{unit}"] = result.get(f"explore_{unit}", 0) + result.get(f"go_{unit}", 0)
        return result

    def close(self): self.db.close()


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
        try: self.env.close()
        finally:
            if self.ledger: self.ledger.close()
