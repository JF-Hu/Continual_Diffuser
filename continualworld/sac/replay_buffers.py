import random
from typing import Dict

import numpy as np
import tensorflow as tf


class ReplayBuffer:
    """A simple FIFO experience replay buffer for SAC agents."""

    def __init__(self, obs_dim: int, act_dim: int, size: int, buffer_name="medium-expert") -> None:
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.actions_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rewards_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.timeout_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.buffer_name = buffer_name

    def store(
        self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool, timeout: bool
    ) -> None:
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.actions_buf[self.ptr] = action
        self.rewards_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done
        self.timeout_buf[self.ptr] = timeout
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def store_multiple(
            self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_obs: np.ndarray, done: np.ndarray, timeout: np.ndarray,
    ) -> None:
        assert len(obs) == len(actions) == len(rewards) == len(next_obs) == len(done) == len(timeout)
        assert self.size + len(obs) <= self.max_size

        range_start = self.size
        range_end = self.size + len(obs)
        self.obs_buf[range_start:range_end] = obs
        self.next_obs_buf[range_start:range_end] = next_obs
        self.actions_buf[range_start:range_end] = actions
        self.rewards_buf[range_start:range_end] = rewards
        self.done_buf[range_start:range_end] = done
        self.timeout_buf[range_start:range_end] = timeout
        self.size = self.size + len(obs)
        self.ptr = (self.ptr + len(obs)) % self.max_size

    def sample_batch(self, batch_size: int) -> Dict[str, tf.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=tf.convert_to_tensor(self.obs_buf[idxs]),
            next_obs=tf.convert_to_tensor(self.next_obs_buf[idxs]),
            actions=tf.convert_to_tensor(self.actions_buf[idxs]),
            rewards=tf.convert_to_tensor(self.rewards_buf[idxs]),
            done=tf.convert_to_tensor(self.done_buf[idxs]),
            timeout=tf.convert_to_tensor(self.timeout_buf[idxs]),
        )

    def valid_buffer_clip(self):
        self.obs_buf = self.obs_buf[:self.size]
        self.next_obs_buf = self.next_obs_buf[:self.size]
        self.actions_buf = self.actions_buf[:self.size]
        self.rewards_buf = self.rewards_buf[:self.size]
        self.done_buf = self.done_buf[:self.size]
        self.timeout_buf = self.timeout_buf[:self.size]

    def get_data(self, key):
        assert key in ["observation", "next_observation", "action", "reward", "done", "timeout"]
        if key == "observation":
            return np.vstack(self.obs_buf)
        if key == "next_observation":
            return np.vstack(self.next_obs_buf)
        if key == "action":
            return np.vstack(self.actions_buf)
        if key == "reward":
            return np.vstack(self.rewards_buf)
        if key == "done":
            return np.vstack(self.done_buf)
        if key == "timeout":
            return np.vstack(self.timeout_buf)
        raise NotImplementedError

    def get_offline_dataset(self):
        offline_dataset = {
            "observations": self.obs_buf,
            "next_observations":self.next_obs_buf,
            "actions":self.actions_buf,
            "rewards":self.rewards_buf,
            "dones":self.done_buf,
            "timeouts":self.timeout_buf}
        return offline_dataset

class EpisodicMemory:
    """Buffer which does not support overwriting old samples."""

    def __init__(self, obs_dim: int, act_dim: int, size: int) -> None:
        self.reset(size=size, obs_dim=obs_dim, act_dim=act_dim)

    def store_multiple(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
        timeout: np.ndarray,
    ) -> None:
        assert len(obs) == len(actions) == len(rewards) == len(next_obs) == len(done)
        assert self.size + len(obs) <= self.max_size

        range_start = self.size
        range_end = self.size + len(obs)
        self.obs_buf[range_start:range_end] = obs
        self.next_obs_buf[range_start:range_end] = next_obs
        self.actions_buf[range_start:range_end] = actions
        self.rewards_buf[range_start:range_end] = rewards
        self.done_buf[range_start:range_end] = done
        self.timeout_buf[range_start:range_end] = timeout
        self.size = self.size + len(obs)
        self.ptr = (self.ptr + len(obs)) % self.max_size

    def store(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool, timeout: bool) -> None:
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.actions_buf[self.ptr] = action
        self.rewards_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done
        self.timeout_buf[self.ptr] = timeout
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int) -> Dict[str, tf.Tensor]:
        batch_size = min(batch_size, self.size)
        idxs = np.random.choice(self.size, size=batch_size, replace=False)
        return dict(
            obs=tf.convert_to_tensor(self.obs_buf[idxs]),
            next_obs=tf.convert_to_tensor(self.next_obs_buf[idxs]),
            actions=tf.convert_to_tensor(self.actions_buf[idxs]),
            rewards=tf.convert_to_tensor(self.rewards_buf[idxs]),
            done=tf.convert_to_tensor(self.done_buf[idxs]),
            timeout=tf.convert_to_tensor(self.timeout_buf[idxs]),
        )

    def reset(self, size, obs_dim, act_dim):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.actions_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rewards_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.timeout_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size


class ReservoirReplayBuffer(ReplayBuffer):
    """Buffer for SAC agents implementing reservoir sampling."""

    def __init__(self, obs_dim: int, act_dim: int, size: int) -> None:
        super().__init__(obs_dim, act_dim, size)
        self.timestep = 0

    def store(
        self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool, timeout: bool
    ) -> None:
        current_t = self.timestep
        self.timestep += 1

        if current_t < self.max_size:
            buffer_idx = current_t
        else:
            buffer_idx = random.randint(0, current_t)
            if buffer_idx >= self.max_size:
                return

        self.obs_buf[buffer_idx] = obs
        self.next_obs_buf[buffer_idx] = next_obs
        self.actions_buf[buffer_idx] = action
        self.rewards_buf[buffer_idx] = reward
        self.done_buf[buffer_idx] = done
        self.timeout_buf[buffer_idx] = timeout
        self.size = min(self.size + 1, self.max_size)

class Episodicbuffer:

    def __init__(self, obs_dim: int, act_dim: int, size: int) -> None:
        self.size = size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs_buf = []
        self.next_obs_buf = []
        self.actions_buf = []
        self.rewards_buf = []
        self.done_buf = []
        self.ptr, self.size, self.max_size = 0, 0, size

    def store_multiple(
        self,
        obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_obs: np.ndarray, done: np.ndarray,
    ) -> None:
        assert len(obs) == len(actions) == len(rewards) == len(next_obs) == len(done)
        assert self.size + len(obs) <= self.max_size

        range_start = self.size
        range_end = self.size + len(obs)
        self.obs_buf.append(obs)
        self.next_obs_buf.append(next_obs)
        self.actions_buf.append(actions)
        self.rewards_buf.append(rewards)
        self.done_buf.append(done)
        self.size = self.size + len(obs)
        self.ptr = (self.ptr + len(obs)) % self.max_size

    def store(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.obs_buf.append(obs)
        self.next_obs_buf.append(next_obs)
        self.actions_buf.append(action)
        self.rewards_buf.append(reward)
        self.done_buf.append(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def reset(self):
        self.obs_buf.clear()
        self.next_obs_buf.clear()
        self.actions_buf.clear()
        self.rewards_buf.clear()
        self.done_buf.clear()
        self.ptr, self.size, self.max_size = 0, 0, self.size

    def get_data(self, key):
        assert key in ["observation", "next_observation", "action", "reward", "done"]
        if key == "observation":
            return np.vstack(self.obs_buf)
        if key == "next_observation":
            return np.vstack(self.next_obs_buf)
        if key == "action":
            return np.vstack(self.actions_buf)
        if key == "reward":
            return np.vstack(self.rewards_buf)
        if key == "done":
            return np.vstack(self.done_buf)
        raise NotImplementedError