import pickle
# from continualworld.envs import get_mt50
import h5py


def load_offline_datasets(project_path, env_name, task_idx, continual_world_dataset_quality="expert", continual_world_data_type="hdf5"):
    if env_name=="cheetah_vel":
        raise NotImplementedError
    elif env_name=="ant_dir":
        assert task_idx in ["4", "6", "7", "9", "10", "13", "15", "16", "17", "18", "19", "21", "22", "23", "24", "25", "26", "27", "28", "29",
                            "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49"]
        dataset_path = f"{project_path}/data/{env_name}/{env_name}-{task_idx}-expert.pkl"
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        return trajectories
    elif env_name == "cheetah_dir":
        assert task_idx in ["0", "1"]
        dataset_path = f"{project_path}/data/{env_name}/{env_name}-{task_idx}-expert.pkl"
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        return trajectories
    elif env_name == "ML1-pick-place-v2":
        assert task_idx in ["3", "4", "5", "6", "7", "8", "28", "29", "35", "37", "38", "42", "43", "45", "46", "47", "48", "49"]
        dataset_path = f"{project_path}/data/{env_name}/{env_name}-{task_idx}-expert.pkl"
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        return trajectories
    elif env_name == "continual_world":
        assert task_idx in [
            "hammer-v1", "push-wall-v1", "faucet-close-v1", "push-back-v1", "stick-pull-v1", "handle-press-side-v1", "push-v1", "shelf-place-v1", "window-close-v1", "peg-unplug-side-v1",
            "pick-place-v1", "door-open-v1", "drawer-open-v1", "drawer-close-v1", "button-press-topdown-v1", "peg-insert-side-v1", "window-open-v1", "door-close-v1", "reach-wall-v1",
            "pick-place-wall-v1", "button-press-v1", "button-press-topdown-wall-v1", "button-press-wall-v1", "disassemble-v1", "plate-slide-v1", "plate-slide-side-v1", "plate-slide-back-v1",
            "plate-slide-back-side-v1", "handle-press-v1", "handle-pull-v1", "handle-pull-side-v1", "stick-push-v1", "basketball-v1", "soccer-v1", "faucet-open-v1", "coffee-push-v1",
            "coffee-pull-v1", "coffee-button-v1", "sweep-v1", "sweep-into-v1", "pick-out-of-hole-v1", "assembly-v1", "push-back-v1", "lever-pull-v1", "dial-turn-v1"]
        dataset_path = f"{project_path}/data/{env_name}/{task_idx}/{task_idx}-{continual_world_dataset_quality}.{continual_world_data_type}"
        if continual_world_data_type == "hdf5":
            file = h5py.File(dataset_path, "r")
            trajectories_keys = ["observations", "next_observations", "actions", "rewards", "dones", "timeouts"]
            trajectories = {}
            for data_key in trajectories_keys:
                data = file[data_key][:]
                if data_key == "dones":
                    data_key = "terminals"
                trajectories[data_key] = data
            file.close()
        elif continual_world_data_type == "pkl":
            file = open(dataset_path, "rb")
            trajectories = pickle.load(file)
            trajectories["terminals"] = trajectories["dones"]
            file.close()
        else:
            raise NotImplementedError
        return trajectories
    else:
        raise NotImplementedError



