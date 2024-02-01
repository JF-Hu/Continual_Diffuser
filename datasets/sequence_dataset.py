import torch
import numpy as np
from datasets.multi_mujoco_env.load_env import load_environment
from datasets.multi_mujoco_env.load_offline_dataset import load_offline_datasets
from path_process.get_path import get_project_path
from datasets.buffer import ReturnReplayBuffer
from datasets.normalizer import DatasetNormalizer
from collections import namedtuple
from datasets.consecutive_traj_2_separate_traj import consecutive_trajectory_2_separate_trajectory, consecutive_trajectory_2_separate_success_trajectory, get_save2buffer_flag
from termcolor import colored
import time

Batch = namedtuple('Batch', 'trajectories conditions observations actions rewards next_observations returns dones')
TaskCondBatch = namedtuple('Batch', 'trajectories conditions task_identity observations actions rewards next_observations returns dones')

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, argus=None, project_path=None, task_idx=4, env_name='ant_dir', sequence_length=64,
        normalizer='GaussianNormalizer', termination_penalty=0, discount=0.99, returns_scale=1000):
        if project_path is None:
            project_path = get_project_path()
        self.project_path = project_path
        self.argus = argus
        self.env_name = env_name
        self.task_idx = task_idx
        self.env = load_environment(project_path=project_path, env_name=env_name, task_idx=task_idx)
        self.eval_env = load_environment(project_path=project_path, env_name=env_name, task_idx=task_idx)
        self.max_path_length = self.env._max_episode_steps
        self.returns_scale = returns_scale
        self.sequence_length = sequence_length
        self.discount = discount
        self.termination_penalty = termination_penalty
        self.replay_buffer = self.get_data_from_dataset(project_path=project_path, env_name=env_name, task_idx=task_idx)
        self.normalizer = DatasetNormalizer(self.replay_buffer, normalizer, path_lengths=self.replay_buffer['path_lengths'])
        self.indices = self.make_indices(self.replay_buffer.path_lengths)
        self.observation_dim = self.replay_buffer.observations[0].shape[-1]
        self.action_dim = self.replay_buffer.actions[0].shape[-1]
        self.transition_dim = self.observation_dim + self.action_dim
        self.n_episodes = self.replay_buffer.n_episodes
        self.path_lengths = self.replay_buffer.path_lengths
        self.max_path_length = np.max(self.replay_buffer.path_lengths)
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.normalize()
        print(self.replay_buffer)

    def get_data_from_dataset(self, project_path, env_name, task_idx):
        trajectories = load_offline_datasets(project_path=project_path, env_name=env_name, task_idx=task_idx)
        replay_buffer = ReturnReplayBuffer(argus=self.argus, termination_penalty=self.termination_penalty, discounts=self.discount, max_path_length=self.max_path_length)
        if env_name == "cheetah_vel":
            raise NotImplementedError
        elif env_name == "ant_dir":
            for episode in trajectories:
                replay_buffer.add_path(episode)
        else:
            raise NotImplementedError
        replay_buffer.finalize()
        return replay_buffer

    def get_max_min_discounted_return(self):
        return np.max(np.vstack(self.replay_buffer._dict['discounted_returns'])), np.min(np.vstack(self.replay_buffer._dict['discounted_returns']))

    def normalize(self, keys=['observations', 'actions', 'next_observations']):
        '''
            'fft_observations'
            normalize fields that will be predicted by the diffusion model
        '''
        if self.env_name.split("-")[0] in ["hammer", "pen", "relocate", "door"]:
            keys = ['observations', 'actions']
        for key in keys:
            self.replay_buffer[f'normed_{key}'] = []
            for path_i, path in enumerate(self.replay_buffer[key]):
                self.replay_buffer[f'normed_{key}'].append(self.normalizer(path, key))

    def make_indices(self, path_lengths):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length, path_length - self.sequence_length)
            for start in range(max_start+1):
                end = start + self.sequence_length
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        if self.argus.train_with_normed_data:
            observations = self.replay_buffer.normed_observations[path_ind][start:end]
            next_observations = self.replay_buffer.normed_next_observations[path_ind][start:end]
            actions = self.replay_buffer.normed_actions[path_ind][start:end]
        else:
            observations = self.replay_buffer.observations[path_ind][start:end]
            next_observations = self.replay_buffer.next_observations[path_ind][start:end]
            actions = self.replay_buffer.actions[path_ind][start:end]
        trajectories = np.concatenate([observations, actions], axis=-1)
        conditions = self.get_conditions(observations)
        rewards = self.replay_buffer.rewards[path_ind][start:end]
        returns = self.replay_buffer.discounted_returns[path_ind][start:end]
        dones = self.replay_buffer.terminals[path_ind][start:end]
        batch = Batch(trajectories, conditions, observations, actions, rewards, next_observations, returns, dones)
        return batch

class MultiTaskSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, task_idx_list, argus=None, project_path=None, env_name='ant_dir', sequence_length=64,
        normalizer='GaussianNormalizer', termination_penalty=0, discount=0.99, returns_scale=1000):
        if project_path is None:
            project_path = get_project_path()
        self.project_path = project_path
        self.argus = argus
        self.env_name = env_name
        self.task_idx_list = task_idx_list
        self.task_labels = self.get_task_labels(env_name=env_name, task_idx_list=task_idx_list)
        self.current_task_idx = task_idx_list[0]
        self.envs, self.eval_envs, self.task_identity, self.action_ranges = {}, {}, {}, {}
        for key_idx, task_idx in enumerate(task_idx_list):
            env, task_identify = load_environment(project_path=project_path, env_name=env_name, task_idx=task_idx, need_task_identify=True)
            self.envs.update({self.task_labels[key_idx]: env})
            self.task_identity.update({self.task_labels[key_idx]: task_identify})
            # env, task_identify = load_environment(project_path=project_path, env_name=env_name, task_idx=task_idx)
            self.eval_envs.update({self.task_labels[key_idx]: [load_environment(project_path=project_path, env_name=env_name, task_idx=task_idx) for _ in range(argus.eval_episodes)]})
        self.max_path_length = self.envs[self.task_labels[0]]._max_episode_steps
        self.returns_scale = returns_scale
        self.sequence_length = sequence_length
        self.discount = discount
        self.termination_penalty = termination_penalty
        self.replay_buffers, self.normalizers, self.tasks_indices, self.tasks_n_episodes, self.tasks_path_lengths = {}, {}, {}, {}, {}
        for key_idx, task_idx in enumerate(task_idx_list):
            self.replay_buffers.update({self.task_labels[key_idx]: self.get_data_from_dataset(project_path=project_path, env_name=env_name, task_idx=task_idx)})
            if self.argus.fix_normalizer_id:
                self.normalizers.update({self.task_labels[key_idx]: DatasetNormalizer(self.replay_buffers[self.task_labels[0]], normalizer, path_lengths=self.replay_buffers[self.task_labels[0]]['path_lengths'])})
            else:
                self.normalizers.update({self.task_labels[key_idx]: DatasetNormalizer(self.replay_buffers[self.task_labels[key_idx]], normalizer, path_lengths=self.replay_buffers[self.task_labels[key_idx]]['path_lengths'])})
            self.tasks_indices.update({self.task_labels[key_idx]: self.make_indices(self.replay_buffers[self.task_labels[key_idx]].path_lengths)})
            self.tasks_n_episodes.update({self.task_labels[key_idx]: self.replay_buffers[self.task_labels[key_idx]].n_episodes})
            self.tasks_path_lengths.update({self.task_labels[key_idx]: self.replay_buffers[self.task_labels[key_idx]].path_lengths})
        self.observation_dim = self.replay_buffers[self.task_labels[0]].observations[0].shape[-1]
        self.action_dim = self.replay_buffers[self.task_labels[0]].actions[0].shape[-1]
        self.task_identity_dim = len(np.reshape(np.array(self.task_identity[self.task_labels[0]]), [-1,]))
        self.transition_dim = self.observation_dim + self.action_dim
        self.max_path_length = np.max(np.concatenate(list(self.tasks_path_lengths.values())))
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.normalize()
        self.action_ranges = self.get_task_specific_action_ranges()
        # print(self.replay_buffers)

    def get_task_specific_action_ranges(self):
        action_ranges = {}
        if self.argus.train_with_normed_data:
            for key_idx, task_idx in enumerate(self.task_idx_list):
                if self.argus.fix_normalizer_id:
                    action_ranges.update({self.task_labels[key_idx]: (np.min(np.vstack(self.replay_buffers[self.task_labels[0]].normed_actions)), np.max(np.vstack(self.replay_buffers[self.task_labels[0]].normed_actions)))})
                else:
                    action_ranges.update({self.task_labels[key_idx]: (np.min(np.vstack(self.replay_buffers[self.task_labels[key_idx]].normed_actions)), np.max(np.vstack(self.replay_buffers[self.task_labels[key_idx]].normed_actions)))})
        else:
            for key_idx, task_idx in enumerate(self.task_idx_list):
                action_ranges.update({self.task_labels[key_idx]: (-0.99999, 0.99999)})
        return action_ranges

    def get_task_labels(self, env_name, task_idx_list):
        task_labels = {}
        if env_name in ["cheetah_vel", "ant_dir", "cheetah_dir", "ML1-pick-place-v2"]:
            for idx, task_idx in enumerate(task_idx_list):
                task_labels[idx] = f"task{task_idx}"
                task_labels[task_idx] = f"task{task_idx}"
        elif env_name == "continual_world":
            for idx, task_idx in enumerate(task_idx_list):
                task_labels[idx] = f"{task_idx}"
                task_labels[task_idx] = f"{task_idx}"
        else:
            raise NotImplementedError
        return task_labels

    def set_task_idx(self, task_idx):
        assert task_idx in self.task_idx_list
        self.current_task_idx = task_idx

    def get_data_from_dataset(self, project_path, env_name, task_idx):
        trajectories = load_offline_datasets(project_path=project_path, env_name=env_name, task_idx=task_idx, continual_world_dataset_quality=self.argus.continual_world_dataset_quality, continual_world_data_type=self.argus.continual_world_data_type)
        replay_buffer = ReturnReplayBuffer(argus=self.argus, termination_penalty=self.termination_penalty, discounts=self.discount, max_path_length=self.max_path_length)
        if env_name == "cheetah_vel":
            raise NotImplementedError
        elif env_name == "ant_dir":
            for episode in trajectories:
                replay_buffer.add_path(episode)
        elif env_name == "cheetah_dir":
            for episode in trajectories:
                replay_buffer.add_path(episode)
        elif env_name == "ML1-pick-place-v2":
            for episode in trajectories:
                replay_buffer.add_path(episode)
        elif env_name == "continual_world":
            if self.argus.continual_world_data_type == "pkl":
                itr = consecutive_trajectory_2_separate_trajectory(dataset=trajectories, max_episode_steps=self.max_path_length, need_ep_range=(self.argus.dataset_load_min, self.argus.dataset_load_max))
                for i, episode in enumerate(itr):
                    if episode is not None and get_save2buffer_flag(env_name=env_name, task_idx=task_idx, episode_returns=sum(episode["rewards"])):
                        assert len(episode["rewards"]) <= self.max_path_length
                        replay_buffer.add_path(episode)
            elif self.argus.continual_world_data_type == "hdf5":
                itr = consecutive_trajectory_2_separate_trajectory(dataset=trajectories, max_episode_steps=self.max_path_length, need_ep_range=(self.argus.dataset_load_min, self.argus.dataset_load_max))
                for i, episode in enumerate(itr):
                    if episode is not None:
                        replay_buffer.add_path(episode)
            else:
                raise NotImplementedError
            print(colored(f"Loading [{self.argus.continual_world_dataset_quality}] dataset from [{self.env_name}.{task_idx}] !!!!!!", color="red"))
        else:
            raise NotImplementedError
        replay_buffer.finalize()
        print(f'[ datasets/buffer ] Finalized task[{task_idx}] replay buffer | {replay_buffer._count} episodes | average ep return is [{np.mean([np.sum(replay_buffer["rewards"][i]) for i in range(replay_buffer._count)])}]')
        return replay_buffer

    def get_max_min_discounted_return(self, task_idx):
        return np.max(np.vstack(self.replay_buffers[f"task{task_idx}"]._dict['discounted_returns'])), np.min(np.vstack(self.replay_buffers[f"task{task_idx}"]._dict['discounted_returns']))

    def normalize(self, keys=['observations', 'actions', 'next_observations']):
        '''
            'fft_observations'
            normalize fields that will be predicted by the diffusion model
        '''
        if self.env_name.split("-")[0] in ["hammer", "pen", "relocate", "door"]:
            keys = ['observations', 'actions']
        for key_idx, task_idx in enumerate(self.task_idx_list):
            for key in keys:
                self.replay_buffers[self.task_labels[key_idx]][f'normed_{key}'] = []
                for path_i, path in enumerate(self.replay_buffers[self.task_labels[key_idx]][key]):
                    self.replay_buffers[self.task_labels[key_idx]][f'normed_{key}'].append(self.normalizers[self.task_labels[key_idx]](path, key))

    def make_indices(self, path_lengths):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length, path_length - self.sequence_length)
            for start in range(max_start+1):
                end = start + self.sequence_length
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        if len(np.shape(observations)) == 2:
            return {0: observations[0]}
        elif len(np.shape(observations)) == 3:
            return {0: observations[:, 0, :]}
        else:
            raise NotImplementedError

    def get_rehearsal_buffer_from_specific_task(self, batch_size, task_name, sample_bound=0.1):
        data_info = np.random.choice(np.maximum(int(len(self.tasks_indices[task_name])*sample_bound), batch_size*2), size=batch_size, replace=False)
        trajectories, conditions, task_identity, observations, actions, rewards, next_observations, returns, dones = [], [], [], [], [], [], [], [], []
        for data_i in data_info:
            data_i = - data_i - 1
            path_ind, start, end = self.tasks_indices[task_name][data_i]
            if self.argus.train_with_normed_data:
                observations.append(np.expand_dims(self.replay_buffers[task_name].normed_observations[path_ind][start:end], axis=0))
                next_observations.append(np.expand_dims(self.replay_buffers[task_name].normed_next_observations[path_ind][start:end], axis=0))
                actions.append(np.expand_dims(self.replay_buffers[task_name].normed_actions[path_ind][start:end], axis=0))
            else:
                observations.append(np.expand_dims(self.replay_buffers[task_name].observations[path_ind][start:end], axis=0))
                next_observations.append(np.expand_dims(self.replay_buffers[task_name].next_observations[path_ind][start:end], axis=0))
                actions.append(np.expand_dims(self.replay_buffers[task_name].actions[path_ind][start:end], axis=0))
            rewards.append(np.expand_dims(self.replay_buffers[task_name].rewards[path_ind][start:end], axis=0))
            returns.append(np.expand_dims(self.replay_buffers[task_name].discounted_returns[path_ind][start], axis=0))
            dones.append(np.expand_dims(self.replay_buffers[task_name].terminals[path_ind][start:end], axis=0))
            task_identity.append(np.reshape(self.task_identity[task_name], [-1, self.task_identity_dim]))
        observations = torch.tensor(np.vstack(observations))
        next_observations = torch.tensor(np.vstack(next_observations))
        actions = torch.tensor(np.vstack(actions))
        trajectories = torch.cat([observations, actions], dim=-1)
        conditions = self.get_conditions(observations)
        rewards = torch.tensor(np.vstack(rewards))
        returns = torch.tensor(np.vstack(returns))
        dones = torch.tensor(np.vstack(dones))
        task_identity = torch.tensor(np.vstack(task_identity))
        batch = TaskCondBatch(trajectories, conditions, task_identity, observations, actions, rewards, next_observations, returns, dones)
        return batch, task_name

    def get_rehearsal_buffer_from_all_previous_tasks(self, batch_size, sample_bound=0.1):
        data_from_task = []
        trajectories, conditions, task_identity, observations, actions, rewards, next_observations, returns, dones = [], [], [], [], [], [], [], [], []
        for task_i_label in range(len(self.task_idx_list)):
            task_name = self.task_labels[task_i_label]
            if self.task_labels[self.current_task_idx] == task_name:
                break
            data_from_task.append(task_name)
            data_info = np.random.choice(np.maximum(int(len(self.tasks_indices[task_name])*sample_bound), batch_size*2), size=batch_size, replace=False)
            for data_i in data_info:
                data_i = - data_i - 1
                path_ind, start, end = self.tasks_indices[task_name][data_i]
                if self.argus.train_with_normed_data:
                    observations.append(np.expand_dims(self.replay_buffers[task_name].normed_observations[path_ind][start:end], axis=0))
                    next_observations.append(np.expand_dims(self.replay_buffers[task_name].normed_next_observations[path_ind][start:end], axis=0))
                    actions.append(np.expand_dims(self.replay_buffers[task_name].normed_actions[path_ind][start:end], axis=0))
                else:
                    observations.append(np.expand_dims(self.replay_buffers[task_name].observations[path_ind][start:end], axis=0))
                    next_observations.append(np.expand_dims(self.replay_buffers[task_name].next_observations[path_ind][start:end], axis=0))
                    actions.append(np.expand_dims(self.replay_buffers[task_name].actions[path_ind][start:end], axis=0))
                rewards.append(np.expand_dims(self.replay_buffers[task_name].rewards[path_ind][start:end], axis=0))
                returns.append(np.expand_dims(self.replay_buffers[task_name].discounted_returns[path_ind][start], axis=0))
                dones.append(np.expand_dims(self.replay_buffers[task_name].terminals[path_ind][start:end], axis=0))
                task_identity.append(np.reshape(self.task_identity[task_name], [-1, self.task_identity_dim]))
        observations = torch.tensor(np.vstack(observations))
        next_observations = torch.tensor(np.vstack(next_observations))
        actions = torch.tensor(np.vstack(actions))
        trajectories = torch.cat([observations, actions], dim=-1)
        conditions = self.get_conditions(observations)
        rewards = torch.tensor(np.vstack(rewards))
        returns = torch.tensor(np.vstack(returns))
        dones = torch.tensor(np.vstack(dones))
        task_identity = torch.tensor(np.vstack(task_identity))
        batch = TaskCondBatch(trajectories, conditions, task_identity, observations, actions, rewards, next_observations, returns, dones)
        return batch, "_".join(data_from_task)

    def __len__(self):
        return len(self.tasks_indices[self.task_labels[self.current_task_idx]])

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.tasks_indices[self.task_labels[self.current_task_idx]][idx]
        if self.argus.train_with_normed_data:
            observations = self.replay_buffers[self.task_labels[self.current_task_idx]].normed_observations[path_ind][start:end]
            next_observations = self.replay_buffers[self.task_labels[self.current_task_idx]].normed_next_observations[path_ind][start:end]
            actions = self.replay_buffers[self.task_labels[self.current_task_idx]].normed_actions[path_ind][start:end]
        else:
            observations = self.replay_buffers[self.task_labels[self.current_task_idx]].observations[path_ind][start:end]
            next_observations = self.replay_buffers[self.task_labels[self.current_task_idx]].next_observations[path_ind][start:end]
            actions = self.replay_buffers[self.task_labels[self.current_task_idx]].actions[path_ind][start:end]
        trajectories = np.concatenate([observations, actions], axis=-1)
        conditions = self.get_conditions(observations)
        rewards = self.replay_buffers[self.task_labels[self.current_task_idx]].rewards[path_ind][start:end]
        returns = self.replay_buffers[self.task_labels[self.current_task_idx]].discounted_returns[path_ind][start]
        dones = self.replay_buffers[self.task_labels[self.current_task_idx]].terminals[path_ind][start:end]
        task_identity = np.reshape(self.task_identity[self.task_labels[self.current_task_idx]], [-1,])
        batch = TaskCondBatch(trajectories, conditions, task_identity, observations, actions, rewards, next_observations, returns, dones)
        return batch

if __name__ == "__main__":
    dataset = SequenceDataset(env_name="ant_dir")
