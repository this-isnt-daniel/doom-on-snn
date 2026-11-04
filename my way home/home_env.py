"""Use Sample Factory's own my-way-home scenario and validate its button mapping."""
from collections import deque
import functools
import hashlib
from pathlib import Path
import numpy as np
import gymnasium as gym
from sample_factory.envs.env_utils import register_env
from sf_examples.vizdoom.doom.doom_utils import DOOM_ENVS, make_doom_env_from_spec


class HomeImages(gym.Wrapper):
    def __init__(self,env,stack):
        super().__init__(env)
        self.stack=int(stack)
        if self.stack<1:raise ValueError("Stack must be positive")
        assert isinstance(env.observation_space,gym.spaces.Box)
        assert env.observation_space.shape==(3,72,128)
        self.frames=deque(maxlen=self.stack)
        self.observation_space=gym.spaces.Dict({"obs":gym.spaces.Box(0.,1.,(3*self.stack,72,128),np.float32)})

    def observation(self):
        return {"obs":np.concatenate(list(self.frames),axis=0).astype(np.float32)/255.}

    def reset(self,**kwargs):
        image,info=self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.stack):self.frames.append(np.asarray(image).copy())
        return self.observation(),info

    def step(self,action):
        image,reward,done,truncated,info=self.env.step(action)
        self.frames.append(np.asarray(image).copy())
        return self.observation(),reward,done,truncated,info


def make_home(spec,name,cfg,env_config,render_mode=None,**kwargs):
    env=make_doom_env_from_spec(spec,name,cfg,env_config,render_mode,**kwargs)
    return HomeImages(env,cfg.home_stack)


def register():
    spec=next(s for s in DOOM_ENVS if s.name=="doom_my_way_home")
    register_env(spec.name,functools.partial(make_home,spec))


def reset_episode(env,seed):
    # Some Sample Factory versions update curr_seed in reset(seed=...), but only
    # call DoomGame.set_seed during initial engine creation. Explicitly seed an
    # already-initialized engine too, before new_episode starts.
    game=env.unwrapped.game
    if game is not None:game.set_seed(int(seed))
    return env.reset(seed=int(seed))


def validate(env,seed):
    """Fail visibly on bundled-asset/action mismatch, rather than silently mislabel controls."""
    assert isinstance(env.action_space,gym.spaces.Tuple)
    assert [s.n for s in env.action_space.spaces]==[3,3],env.action_space
    obs,_=reset_episode(env,seed)
    game=env.unwrapped.game
    buttons=[b.name for b in game.get_available_buttons()]
    expected=["TURN_LEFT","TURN_RIGHT","MOVE_FORWARD","MOVE_BACKWARD"]
    if buttons!=expected:
        raise RuntimeError(f"Sample Factory tuple mapping expects {expected}, but scenario exposes {buttons}. "
                           "Do not substitute ViZDoom's generic cfg: its controls may differ.")
    config=Path(env.unwrapped.config_path)
    return obs,{"buttons":buttons,"scenario_config":str(config),
                "scenario_config_sha256":hashlib.sha256(config.read_bytes()).hexdigest(),
                "episode_timeout_tics":game.get_episode_timeout(),
                "living_reward":game.get_living_reward()}
