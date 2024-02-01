import pickle
from datasets.multi_mujoco_env.mujoco_control_envs import HalfCheetahDirEnv, HalfCheetahVelEnv, AntDirEnv
import metaworld
# from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE, ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
from continualworld.envs import get_single_env

def load_environment(project_path, env_name, task_idx, need_task_identify=False):
    if env_name=="cheetah_vel":
        assert isinstance(task_idx, str) and int(task_idx) >= 0 and int(task_idx) <= 39
        tasks = []
        with open(f"{project_path}/datasets/multi_mujoco_env/cheetah_vel_config/config_cheetah_vel_task{task_idx}.pkl", 'rb') as f:
            task_info = pickle.load(f)
            tasks.append(task_info[0])
        env = HalfCheetahVelEnv(tasks, include_goal=False)
        if need_task_identify:
            return env, None
        else:
            return env
    elif env_name=="ant_dir":
        assert isinstance(task_idx, str) and int(task_idx) >= 0 and int(task_idx) <= 49
        tasks = []
        with open(f"{project_path}/datasets/multi_mujoco_env/ant_dir_config/config_ant_dir_task{task_idx}.pkl", 'rb') as f:
            task_info = pickle.load(f)
            tasks.append(task_info[0])
        env = AntDirEnv(tasks, len(tasks), include_goal = False)
        task_identity = env._goal
        if need_task_identify:
            return env, task_identity
        else:
            return env
    elif env_name == "cheetah_dir":
        assert isinstance(task_idx, str) and int(task_idx) >= 0 and int(task_idx) <= 1
        if task_idx == "0":
            env = HalfCheetahDirEnv([{'direction': 1}], include_goal = False)
            task_identity = 1
        elif task_idx == "1":
            env = HalfCheetahDirEnv([{'direction': -1}], include_goal = False)
            task_identity = -1
        else:
            raise NotImplementedError
        if need_task_identify:
            return env, task_identity
        else:
            return env
    elif env_name in ["ML1-pick-place-v2"]:
        task_identity = task_idx
        task_name = '-'.join(env_name.split('-')[1:])
        ml1 = metaworld.ML1(task_name, seed=1)  # Construct the benchmark, sampling tasks, note: our example datasets also have seed=1.
        env = ml1.train_classes[task_name]()  # Create an environment with task\
        task = ml1.train_tasks[task_idx]
        env.set_task(task)  # Set task
        env._max_episode_steps = env.max_path_length
        if need_task_identify:
            return env, task_identity
        else:
            return env
    elif env_name in ["continual_world"]:
        task_name = task_idx
        env = get_single_env(task_name)
        env._max_episode_steps = 200
        # if task_name in ["drawer-open-v1"]:
        #     task_identity = 0
        # else:
        #     task_identity = env.unwrapped.goal
        task_identity = 0
        if need_task_identify:
            return env, task_identity
        else:
            return env
    else:
        raise NotImplementedError




#  todo  ("peg-insert-side-v2", SawyerPegInsertionSideEnvV2),

