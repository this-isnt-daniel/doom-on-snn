"""Environment and model definitions for the opt-in defend-the-center LIF sweep."""
from collections import deque
import functools
import numpy as np
import torch
from torch import nn
import gymnasium as gym
from vizdoom import GameVariable
from sample_factory.model.encoder import Encoder
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.envs.env_utils import register_env
from sf_examples.vizdoom.doom.doom_utils import DOOM_ENVS, make_doom_env_from_spec
from sweep_lif_base import SpikingTrunk, calibrate_gains


class GameplayInputs(gym.Wrapper):
    """Stack images and optionally expose ammo; shaping is training-only."""
    def __init__(self, env, stack=1, ammo=False, penalty=0.0):
        super().__init__(env)
        self.stack, self.ammo, self.penalty = int(stack), bool(ammo), float(penalty)
        raw = env.observation_space
        assert isinstance(raw, gym.spaces.Box) and raw.shape == (3, 72, 128), raw
        # Scale only pixels here. With obs_scale=1 in Sample Factory, the ammo
        # scalar cannot accidentally be divided by the global image scale.
        spaces = {"obs": gym.spaces.Box(0., 1., (3*self.stack, 72, 128), np.float32)}
        if self.ammo:
            spaces["ammo"] = gym.spaces.Box(0., 1., (1,), np.float32)
        self.observation_space = gym.spaces.Dict(spaces)
        self.frames = deque(maxlen=self.stack)

    def ammo_value(self):
        return float(self.unwrapped.game.get_game_variable(GameVariable.AMMO2))

    def observation(self, image):
        result = {"obs": np.concatenate(list(self.frames), axis=0).astype(np.float32)/255.}
        if self.ammo:
            result["ammo"] = np.array([np.clip(self.ammo_value()/26., 0., 1.)], np.float32)
        return result

    def reset(self, **kwargs):
        image, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.stack): self.frames.append(np.asarray(image).copy())
        return self.observation(image), info

    def step(self, action):
        before = self.ammo_value()
        image, reward, done, truncated, info = self.env.step(action)
        used = max(0., before-self.ammo_value())
        self.frames.append(np.asarray(image).copy())
        info = dict(info, original_reward=float(reward), ammo_consumed=used)
        return self.observation(image), float(reward)-self.penalty*used, done, truncated, info


def make_env(spec, name, cfg, env_config, render_mode=None, **kwargs):
    base = make_doom_env_from_spec(spec, name, cfg, env_config, render_mode, **kwargs)
    return GameplayInputs(base, cfg.sweep_stack, cfg.sweep_ammo,
                          0. if cfg.sweep_eval else cfg.sweep_penalty)


class SweepEncoder(Encoder):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)
        gains = [float(v) for v in cfg.snn_gains.split(",")]
        shape = obs_space["obs"].shape
        self.trunk = SpikingTrunk(shape[0], shape[1:], cfg.snn_T, .5,
                                  gains, cfg.snn_feat_scale, beta=cfg.snn_beta)
        self.with_ammo = cfg.sweep_ammo
        self.output_size = 512 + int(self.with_ammo)

    def forward(self, obs):
        features = self.trunk(obs["obs"])
        if self.with_ammo:
            features = torch.cat([features, obs["ammo"].to(features.dtype)], dim=-1)
        return features

    def get_out_size(self): return self.output_size


def encoder_factory(cfg, obs_space): return SweepEncoder(cfg, obs_space)


def register():
    spec = next(s for s in DOOM_ENVS if s.name == "doom_defend_the_center")
    register_env(spec.name, functools.partial(make_env, spec))
    global_model_factory().register_encoder_factory(encoder_factory)


def add_args(parser):
    from sample_factory.utils.utils import str2bool
    parser.add_argument("--sweep_stack", type=int, default=1)
    parser.add_argument("--sweep_ammo", type=str2bool, default=False)
    parser.add_argument("--sweep_penalty", type=float, default=0.)
    parser.add_argument("--sweep_eval", type=str2bool, default=False)
    parser.add_argument("--snn_T", type=int, default=8)
    parser.add_argument("--snn_beta", type=float, default=.9)
    parser.add_argument("--snn_feat_scale", type=float, default=4.)
    parser.add_argument("--snn_gains", type=str, default="1,1,1,1")
