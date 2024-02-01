
import collections
import numpy as np


def consecutive_trajectory_2_separate_trajectory(dataset, max_episode_steps, need_ep_range=(-1, 99999999), **kwargs):
    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)
    use_timeouts = 'timeouts' in dataset
    use_dones = 'dones' in dataset
    episode_step = 0
    current_ep = 0
    for i in range(N):
        if use_dones:
            done_bool = bool(dataset['dones'][i])
        else:
            done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == max_episode_steps - 1)

        if current_ep > need_ep_range[0] and current_ep < need_ep_range[1]:
            for key in list(dataset.keys()):
                data_[key].append(np.expand_dims(dataset[key][i], axis=0))

        if done_bool or final_timestep:
            if current_ep > need_ep_range[0] and current_ep < need_ep_range[1]:
                episode_data = {}
                for k in data_:
                    episode_data[k] = np.vstack(data_[k])
            else:
                episode_data = None
            yield episode_data
            data_ = collections.defaultdict(list)
            episode_step = 0
            current_ep += 1

        episode_step += 1

def get_save2buffer_flag(env_name, task_idx, episode_returns):
    if env_name == "cheetah_vel":
        return True
    elif env_name == "ant_dir":
        return True
    elif env_name == "cheetah_dir":
        return True
    elif env_name == "ML1-pick-place-v2":
        return True
    elif env_name == "continual_world":
        if task_idx == "push-wall-v1":
            return episode_returns > 0
        elif task_idx == "shelf-place-v1":
            return episode_returns > 0
        else:
            return True
    else:
        raise NotImplementedError


def consecutive_trajectory_2_separate_success_trajectory(dataset, max_episode_steps, need_ep_range=(-1, 99999999), **kwargs):
    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)
    use_timeouts = 'timeouts' in dataset
    use_dones = 'dones' in dataset
    episode_step = 0
    current_ep = 0
    for i in range(N):
        if use_dones:
            done_bool = bool(dataset['dones'][i])
        else:
            done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == max_episode_steps - 1)

        for key in list(dataset.keys()):
            data_[key].append(np.expand_dims(dataset[key][i], axis=0))

        episode_step += 1

        if done_bool or final_timestep:
            episode_step = 0
            if np.sum(data_["successes"]) > 0.5:
                episode_data = {}
                for k in data_:
                    episode_data[k] = np.vstack(data_[k])
            else:
                episode_data = None
            yield episode_data
            data_ = collections.defaultdict(list)
            current_ep += 1

