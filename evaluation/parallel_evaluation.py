import time
import numpy as np
from datasets.multi_mujoco_env.load_env import load_environment
from trainer.trainer_util import to_torch, to_np
import copy


def parallel_ant_dir_eval(argus, dataset, model, eval_task_idx, eval_episodes=30, ddim_sample=True):
    start_time = time.time()
    reward = [0. for _ in range(eval_episodes)]
    avg_reward = [0. for _ in range(eval_episodes)]
    t = [0 for _ in range(eval_episodes)]
    eval_envs = [load_environment(project_path=dataset.project_path, env_name=dataset.env_name, task_idx=eval_task_idx) for _ in range(eval_episodes)]
    for env_i in range(eval_episodes):
        eval_envs[env_i].seed(np.random.randint(0, 999))
    eval_results = {}
    env_info = {"observations": None, "actions": None, "rewards": None, "next_observations": None}
    obs, done = np.vstack([eval_envs[_].reset() for _ in range(eval_episodes)]), [0 for _ in range(eval_episodes)]
    env_info.update({"observations": obs})
    while np.sum(done) < eval_episodes:
        if argus.train_with_normed_data:
            action = to_np(model.get_action(
                cond={0: to_torch(dataset.normalizer.normalize(x=obs, key="observations"), device=argus.device)},
                data_shape=(eval_episodes, argus.sequence_length, argus.input_channels), ddim_sample=ddim_sample))
            action = dataset.normalizer.unnormalize(x=action[:, 0, -argus.action_dim:], key="actions")
        else:
            action = to_np(model.get_action(
                cond={0: to_torch(obs, device=argus.device)},
                data_shape=(eval_episodes, argus.sequence_length, argus.input_channels), ddim_sample=ddim_sample))
            action = action[:, 0, -argus.action_dim:]
        next_obs = copy.deepcopy(obs)
        for env_i in range(eval_episodes):
            if done[env_i] < 0.5:
                obs_i, reward_i, done_i, _ = eval_envs[env_i].step(action[env_i])
                reward[env_i] = reward_i
                avg_reward[env_i] += reward_i
                t[env_i] += 1
                if t[env_i] >= dataset.max_path_length:
                    done_i = 1
                done[env_i] = int(done_i)
                next_obs[env_i] = obs_i
        env_info.update({"actions": action, "rewards": reward, "next_observations": next_obs})
        # next_obs = to_np(next_obs)
        for env_i in range(eval_episodes):
            if done[env_i] > 0.5:
                next_obs[env_i] = obs[env_i]
        obs = next_obs
        if np.max(t) % 20 == 0:
            print(f"Completing {np.max(t)} timestep. Time Consumption: {time.time() - start_time}")
    eval_results.update({f"Ttask{argus.task_idx}_Etask{eval_task_idx}_ave_return": np.mean(avg_reward), f"Ttask{argus.task_idx}_Etask{eval_task_idx}_ave_time_step": np.mean(t)})
    print(f"Time Consumption: {time.time() - start_time}, {eval_results}")
    return eval_results

def get_obs_with_task(env_name, eval_envs, eval_episodes):
    if env_name in ["cheetah_vel", "ant_dir", "cheetah_dir"]:
        obs = np.vstack([eval_envs[_].reset() for _ in range(eval_episodes)])
    elif env_name in ["ML1-pick-place-v2"]:
        obs = np.vstack([eval_envs[_].reset()[0] for _ in range(eval_episodes)])
    elif env_name in ["continual_world"]:
        obs = np.vstack([eval_envs[_].reset() for _ in range(eval_episodes)])
    else:
        raise NotImplementedError
    return obs

def multi_task_parallel_ant_dir_eval(argus, dataset, model, eval_task_idx, eval_episodes=30, ddim_sample=True, fix_normalizer_id=False):
    task_identity = to_torch(np.tile(np.reshape(np.array(dataset.task_identity[dataset.task_labels[eval_task_idx]]), [-1, dataset.task_identity_dim]), [eval_episodes, 1]), device=argus.device)
    start_time = time.time()
    reward = [0. for _ in range(eval_episodes)]
    avg_reward = [0. for _ in range(eval_episodes)]
    t = [0 for _ in range(eval_episodes)]
    # eval_envs = [load_environment(project_path=dataset.project_path, env_name=dataset.env_name, task_idx=eval_task_idx) for _ in range(eval_episodes)]
    eval_envs = dataset.eval_envs[dataset.task_labels[eval_task_idx]]
    for env_i in range(eval_episodes):
        eval_envs[env_i].seed(np.random.randint(0, 999))
    eval_results = {}
    env_info = {"observations": None, "actions": None, "rewards": None, "next_observations": None}
    # obs, done = np.vstack([eval_envs[_].reset() for _ in range(eval_episodes)]), [0 for _ in range(eval_episodes)]
    obs, done = get_obs_with_task(env_name=dataset.env_name, eval_envs=eval_envs, eval_episodes=eval_episodes), [0 for _ in range(eval_episodes)]
    env_info.update({"observations": obs})
    while np.sum(done) < eval_episodes:
        if argus.train_with_normed_data:
            action = to_np(model.get_action(
                cond={0: to_torch(dataset.normalizers[dataset.task_labels[0] if fix_normalizer_id else dataset.task_labels[eval_task_idx]].normalize(x=obs, key="observations"), device=argus.device)},
                task_identity = task_identity, data_shape=(eval_episodes, argus.sequence_length, argus.input_channels), ddim_sample=ddim_sample,
                task_name=dataset.task_labels[eval_task_idx], action_range=argus.action_ranges[dataset.task_labels[eval_task_idx]]))
            action = dataset.normalizers[dataset.task_labels[0] if fix_normalizer_id else dataset.task_labels[eval_task_idx]].unnormalize(x=action[:, 0, -argus.action_dim:], key="actions")
        else:
            action = to_np(model.get_action(
                cond={0: to_torch(obs, device=argus.device)}, task_identity = task_identity, data_shape=(eval_episodes, argus.sequence_length, argus.input_channels),
                ddim_sample=ddim_sample, task_name=dataset.task_labels[eval_task_idx], action_range=argus.action_ranges[dataset.task_labels[eval_task_idx]]))
            action = action[:, 0, -argus.action_dim:]
        action = np.clip(action, -0.9999999, 0.9999999)
        next_obs = copy.deepcopy(obs)
        for env_i in range(eval_episodes):
            if done[env_i] < 0.5:
                if dataset.env_name in ["cheetah_vel", "ant_dir", "cheetah_dir"]:
                    obs_i, reward_i, done_i, _ = eval_envs[env_i].step(action[env_i])
                elif dataset.env_name in ["ML1-pick-place-v2"]:
                    obs_i, reward_i, done_i, _, __ = eval_envs[env_i].step(action[env_i])
                elif dataset.env_name in ["continual_world"]:
                    obs_i, reward_i, done_i, _ = eval_envs[env_i].step(action[env_i])
                else:
                    raise NotImplementedError
                # obs_i, reward_i, done_i, _ = eval_envs[env_i].step(action[env_i])
                reward[env_i] = reward_i
                avg_reward[env_i] += reward_i
                t[env_i] += 1
                if t[env_i] >= dataset.max_path_length:
                    done_i = 1
                done[env_i] = int(done_i)
                next_obs[env_i] = obs_i
        env_info.update({"actions": action, "rewards": reward, "next_observations": next_obs})
        # next_obs = to_np(next_obs)
        for env_i in range(eval_episodes):
            if done[env_i] > 0.5:
                next_obs[env_i] = obs[env_i]
        obs = next_obs
        if np.max(t) % 20 == 0:
            print(f"Completing {np.max(t)} timestep. Time Consumption: {time.time() - start_time}")
    eval_results.update({f"Ttask{dataset.current_task_idx}_Etask{eval_task_idx}_ave_return": np.mean(avg_reward)})
    print(f"Time Consumption: {time.time() - start_time}, {eval_results}")
    return eval_results

def multi_task_parallel_continual_world_eval(argus, dataset, model, eval_task_idx, eval_episodes=30, ddim_sample=True, fix_normalizer_id=False):
    task_identity = to_torch(np.tile(np.reshape(np.array(dataset.task_identity[dataset.task_labels[eval_task_idx]]), [-1, dataset.task_identity_dim]), [eval_episodes, 1]), device=argus.device)
    start_time = time.time()
    success_rate = [0. for _ in range(eval_episodes)]
    reward = [0. for _ in range(eval_episodes)]
    avg_reward = [0. for _ in range(eval_episodes)]
    t = [0 for _ in range(eval_episodes)]
    eval_envs = dataset.eval_envs[dataset.task_labels[eval_task_idx]]
    for env_i in range(eval_episodes):
        eval_envs[env_i].seed(np.random.randint(0, 999))
        # eval_envs[env_i].seed(0)
    eval_results = {}
    env_info = {"observations": None, "actions": None, "rewards": None, "next_observations": None}
    # obs, done = np.vstack([eval_envs[_].reset() for _ in range(eval_episodes)]), [0 for _ in range(eval_episodes)]
    obs, done = get_obs_with_task(env_name=dataset.env_name, eval_envs=eval_envs, eval_episodes=eval_episodes), [0 for _ in range(eval_episodes)]
    env_info.update({"observations": obs})
    while np.sum(done) < eval_episodes:
        if argus.train_with_normed_data:
            action = to_np(model.get_action(
                cond={0: to_torch(dataset.normalizers[dataset.task_labels[0] if fix_normalizer_id else dataset.task_labels[eval_task_idx]].normalize(x=obs, key="observations"), device=argus.device)},
                task_identity=task_identity, data_shape=(eval_episodes, argus.sequence_length, argus.input_channels), ddim_sample=ddim_sample,
                task_name=dataset.task_labels[eval_task_idx], action_range=argus.action_ranges[dataset.task_labels[eval_task_idx]]))
            action = dataset.normalizers[dataset.task_labels[0] if fix_normalizer_id else dataset.task_labels[eval_task_idx]].unnormalize(x=action[:, 0, -argus.action_dim:], key="actions")
        else:
            action = to_np(model.get_action(
                cond={0: to_torch(obs, device=argus.device)}, task_identity = task_identity, data_shape=(eval_episodes, argus.sequence_length, argus.input_channels),
                ddim_sample=ddim_sample, task_name=dataset.task_labels[eval_task_idx], action_range=argus.action_ranges[dataset.task_labels[eval_task_idx]]))
            action = action[:, 0, -argus.action_dim:]
        action = np.clip(action, -0.9999999, 0.9999999)
        next_obs = copy.deepcopy(obs)
        for env_i in range(eval_episodes):
            if done[env_i] < 0.5:
                if dataset.env_name in ["cheetah_vel", "ant_dir", "cheetah_dir"]:
                    obs_i, reward_i, done_i, info = eval_envs[env_i].step(action[env_i])
                elif dataset.env_name in ["ML1-pick-place-v2"]:
                    obs_i, reward_i, done_i, _, info = eval_envs[env_i].step(action[env_i])
                elif dataset.env_name in ["continual_world"]:
                    obs_i, reward_i, done_i, info = eval_envs[env_i].step(action[env_i])
                else:
                    raise NotImplementedError
                # obs_i, reward_i, done_i, _ = eval_envs[env_i].step(action[env_i])
                reward[env_i] = reward_i
                avg_reward[env_i] += reward_i
                t[env_i] += 1
                if t[env_i] >= dataset.max_path_length or info["success"]:
                    done_i = 1
                    success_rate[env_i] = int(info["success"])
                done[env_i] = int(done_i)
                next_obs[env_i] = obs_i
        env_info.update({"actions": action, "rewards": reward, "next_observations": next_obs})
        # next_obs = to_np(next_obs)
        for env_i in range(eval_episodes):
            if done[env_i] > 0.5:
                next_obs[env_i] = obs[env_i]
        obs = next_obs
        if np.max(t) % 20 == 0:
            print(f"Completing {np.max(t)} timestep. Time Consumption: {time.time() - start_time}")
    # for env_i in range(eval_episodes):
    #     success_rate.append(eval_envs[env_i].current_success)
    eval_results.update({f"Ttask{dataset.current_task_idx}_Etask{eval_task_idx}_ave_return": np.mean(avg_reward), f"Ttask{dataset.current_task_idx}_Etask{eval_task_idx}_success_rate": np.mean(success_rate)})
    print(f"Time Consumption: {time.time() - start_time}, {eval_results}")
    return eval_results