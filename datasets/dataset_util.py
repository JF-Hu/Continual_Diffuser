




def get_multi_task_name(argus):
    postfix, prefix = [], []
    multi_task_name = []
    for task_id in argus.task_idx_list:
        task_id = (str(task_id)).lower()
        if argus.dataset in ["cheetah_vel", "ant_dir", "cheetah_dir", "ML1-pick-place-v2"]:
            new_task_id = task_id
            if len(prefix) == 0:
                prefix.append("task")
        elif argus.dataset == "continual_world":
            task_id = task_id.split("-")
            new_task_id = []
            for item in task_id:
                if "v1" in item or "v2" in item:
                    if len(postfix) == 0:
                        postfix.append(item)
                else:
                    new_task_id.append(f"{(item[0]).upper()}{item[1:3]}")
            new_task_id = "".join(new_task_id)
        else:
            raise NotImplementedError
        multi_task_name.append(new_task_id)
    if len(prefix) > 0:
        multi_task_name.insert(0, prefix[0])
    if len(postfix) > 0:
        multi_task_name.append(postfix[0])
    multi_task_name = "-".join(multi_task_name)
    if "-".join(argus.task_idx_list) == "-".join(["hammer-v1", "push-wall-v1", "faucet-close-v1", "push-back-v1", "stick-pull-v1", "handle-press-side-v1", "push-v1", "shelf-place-v1", "window-close-v1", "peg-unplug-side-v1", "hammer-v1", "push-wall-v1", "faucet-close-v1", "push-back-v1", "stick-pull-v1", "handle-press-side-v1", "push-v1", "shelf-place-v1", "window-close-v1", "peg-unplug-side-v1"]):
        multi_task_name = "cw20"
    # if "-".join(argus.task_idx_list) == "-".join(["hammer-v1", "push-wall-v1", "faucet-close-v1", "push-back-v1", "stick-pull-v1", "handle-press-side-v1", "push-v1", "shelf-place-v1", "window-close-v1", "peg-unplug-side-v1"]):
    #     multi_task_name = "cw10"
    return multi_task_name