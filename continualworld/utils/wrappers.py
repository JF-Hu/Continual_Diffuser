import random
from typing import Any, Dict, List, Tuple

import gym
import metaworld
import numpy as np
from gym.spaces import Box


def recursive_env_name_search(env):
    if hasattr(env, "unwrapped"):
        return recursive_env_name_search(env)
    else:
        return env

class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            # done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

class EnvVersionObsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env) 
        self.name = self.env.name

    def reset(self, **kwargs) -> np.ndarray:
        obs = self.env.reset(**kwargs)
        if "v2" in self.name:
            obs = obs[0]
        return obs

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        if "v1" in self.name:
            obs, reward, done, info = self.env.step(action)
        elif "v2" in self.name:
            obs, reward, done, _, info = self.env.step(action)
        else:
            raise NotImplementedError
        return obs, reward, done, info

class SuccessCounter(gym.Wrapper):
    """Helper class to keep count of successes in MetaWorld environments."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        try:
            self.name = self.env.name
        except:
            self.name = "no specific name"
        self.successes = []
        self.current_success = False

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        obs, reward, done, info = self.env.step(action)
        if info.get("success", False):
            self.current_success = True
        if done:
            self.successes.append(self.current_success)
        return obs, reward, self.current_success, info

    def pop_successes(self) -> List[bool]:
        res = self.successes
        self.successes = []
        return res

    def reset(self, **kwargs) -> np.ndarray:
        self.current_success = False
        return self.env.reset(**kwargs)


class OneHotAdder(gym.Wrapper):
    """Appends one-hot encoding to the observation. Can be used e.g. to encode the task."""

    def __init__(
        self, env: gym.Env, one_hot_idx: int, one_hot_len: int, orig_one_hot_dim: int = 0
    ) -> None:
        super().__init__(env)
        try:
            self.name = self.env.name
        except:
            self.name = "no specific name"

        assert 0 <= one_hot_idx < one_hot_len
        self.to_append = np.zeros(one_hot_len)
        self.to_append[one_hot_idx] = 1.0

        orig_obs_low = self.env.observation_space.low
        orig_obs_high = self.env.observation_space.high
        if orig_one_hot_dim > 0:
            orig_obs_low = orig_obs_low[:-orig_one_hot_dim]
            orig_obs_high = orig_obs_high[:-orig_one_hot_dim]
        self.observation_space = Box(
            np.concatenate([orig_obs_low, np.zeros(one_hot_len)]),
            np.concatenate([orig_obs_high, np.ones(one_hot_len)]),
        )
        self.orig_one_hot_dim = orig_one_hot_dim

    def _append_one_hot(self, obs: np.ndarray) -> np.ndarray:
        if self.orig_one_hot_dim > 0:
            obs = obs[: -self.orig_one_hot_dim]
        return np.concatenate([obs, self.to_append])

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        obs, reward, done, info = self.env.step(action)
        return self._append_one_hot(obs), reward, done, info

    def reset(self, **kwargs) -> np.ndarray:
        return self._append_one_hot(self.env.reset(**kwargs))


class RandomizationWrapper(gym.Wrapper):
    """Manages randomization settings in MetaWorld environments."""

    ALLOWED_KINDS = [
        "deterministic",
        "random_init_all",
        "random_init_fixed20",
        "random_init_small_box",
    ]

    def __init__(self, env: gym.Env, subtasks: List[metaworld.Task], kind: str) -> None:
        assert kind in RandomizationWrapper.ALLOWED_KINDS
        super().__init__(env)
        self.subtasks = subtasks
        self.kind = kind
        try:
            self.name = self.env.name
        except:
            self.name = "no specific name"
        env.set_task(subtasks[0])
        if kind == "random_init_all":
            env._freeze_rand_vec = False

        if kind == "random_init_fixed20":
            assert len(subtasks) >= 20

        if kind == "random_init_small_box":
            diff = env._random_reset_space.high - env._random_reset_space.low
            self.reset_space_low = env._random_reset_space.low + 0.45 * diff
            self.reset_space_high = env._random_reset_space.low + 0.55 * diff

    def reset(self, **kwargs) -> np.ndarray:
        if self.kind == "random_init_fixed20":
            self.env.set_task(self.subtasks[random.randint(0, 19)])
        elif self.kind == "random_init_small_box":
            rand_vec = np.random.uniform(
                self.reset_space_low, self.reset_space_high, size=self.reset_space_low.size
            )
            self.env._last_rand_vec = rand_vec

        return self.env.reset(**kwargs)




class SuccessWrapper(gym.Wrapper):
    """Helper class to keep count of successes in MetaWorld environments."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        # self.env = env.unwrapped
        self.current_success = False

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        obs, reward, done, info = self.env.step(action)
        if info.get("success", False):
            self.current_success = True
        return obs, reward, done, info

    def reset(self, **kwargs) -> np.ndarray:
        self.current_success = False
        return self.env.reset(**kwargs)

    def is_success(self):
        return self.current_success

    def get_current_task_name(self):
        return self.name
