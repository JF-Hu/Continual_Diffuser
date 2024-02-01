import os
import copy
import numpy as np
import torch
import einops
import wandb
import time
import random
import pickle
from termcolor import colored
from ml_logger import logger
from config.dict2class import obj2dict
from .trainer_util import (
    batch_to_device,
    to_np,
    to_device,
    apply_dict,
)
from evaluation.parallel_evaluation import multi_task_parallel_ant_dir_eval, multi_task_parallel_continual_world_eval
from config.hyperparameter import FineTuneParaType, UnetType
from datasets.dataset_util import get_multi_task_name

def cycle(dl):
    while True:
        for data in dl:
            yield data

def cycle_dataloader(argus, dataset, train_batch_size):
    random.seed(argus.seed)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True)
    while True:
        for data in dataset_loader:
            yield data
        print("Finish this epoch dataloader !!!!!!!")
        random.seed(np.random.randint(0, 9999))
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True)
        random.seed(argus.seed)

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class TaskCondTrainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        argus
    ):
        super().__init__()
        self.argus = argus
        self.model = diffusion_model
        self.dataset = dataset
        self.device = argus.device
        self.model.to(argus.device)
        self.ema = EMA(argus.ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = argus.dm_update_ema_every
        self.step_start_ema = argus.dm_step_start_ema
        self.log_freq = argus.dm_log_freq
        self.save_freq = argus.dm_save_freq
        self.batch_size = argus.dm_batch_size
        self.gradient_accumulate_every = argus.dm_gradient_accumulate_every
        self.optimizer = None
        self.base_model_optimizer = self.generate_base_model_opt() #torch.optim.Adam(diffusion_model.parameters(), lr=argus.dm_lr)
        self.LoRA_opts = self.generate_LoRA_opts()
        self.wandb_log = argus.wandb_log
        self.wandb_exp_name = argus.wandb_exp_name
        self.wandb_exp_group = argus.wandb_exp_group
        self.wandb_log_frequency = argus.wandb_log_frequency
        self.wandb_project_name = argus.wandb_project_name
        self.env_name = argus.dataset
        self.current_exp_label = argus.current_exp_label
        self.save_path = argus.save_path
        self.reset_parameters()
        self.step = 0
        self.save_parameter_flag = False
        self.train_on_ith_task = 0
        # self.save_training_parameters()
        if self.wandb_log:
            wandb.init(name=self.wandb_exp_name, group=self.wandb_exp_group, project=self.wandb_project_name, config=obj2dict(self.argus))

    def get_detailed_save_path(self):
        return os.path.join(
            self.save_path, "task_condition", "_".join(self.env_name.split("-")), get_multi_task_name(argus=self.argus), self.argus.current_exp_label)

    def get_lora_model_prefix(self):
        if self.argus.finetune_para_type == FineTuneParaType.ResidualTemporalBlock_TimeMlp:
            lora_model_prefix = "ResidualTemporalBlock_TimeMlp"
        elif self.argus.finetune_para_type == FineTuneParaType.TemporalDowns:
            lora_model_prefix = "TemporalDowns"
        elif self.argus.finetune_para_type == FineTuneParaType.CrossAttention:
            lora_model_prefix = "Spatial_TF_CrossAttention"
        else:
            raise NotImplementedError
        return lora_model_prefix

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def generate_base_model_opt(self):
        parameters = self.model.generate_base_model_parameters(argus=self.argus, list_out=True)
        total_parameters_size = np.sum([para.numel() for para in parameters]) * 4 / 1024 / 1024
        print(colored(f"The size (unit M) of parameters of base model is {total_parameters_size}M", color="green"))
        return torch.optim.Adam(parameters, lr=self.argus.dm_lr)

    def generate_LoRA_opts(self):
        if self.argus.finetune_para_type in [FineTuneParaType.ResidualTemporalBlock_TimeMlp, FineTuneParaType.TemporalDowns,
                                             FineTuneParaType.CrossAttention, FineTuneParaType.InMidOutConv1d]:
            opts = {}
            lora_parameters_size = {}
            for task_name in self.model.model.task_names:
                parameters =  self.model.model.generate_LoRA_parameters(argus=self.argus, task_name=task_name, list_out=True)
                lora_parameters_size[task_name] = np.sum([para.numel() for para in parameters]) * 4 / 1024 / 1024
                opts[task_name] = torch.optim.Adam(parameters, lr=1e-4)
            print(colored(f"The size (unit M) of parameters of lora model is {lora_parameters_size}", color="green"))
            return opts
        elif self.argus.finetune_para_type in [FineTuneParaType.NoFreeze, FineTuneParaType.FullFineTune]:
            return None
        else:
            raise NotImplementedError

    def disable_base_model_training(self):
        self.model.model.control_base_model_training(train_mode=False)

    def enable_lora_training(self, task_name):
        for task in self.model.model.task_names:
            if task == task_name:
                self.model.model.control_lora_model_training(task_name=task, train_mode=True)
            else:
                self.model.model.control_lora_model_training(task_name=task, train_mode=False)

    def switch_optimizer(self):
        if self.argus.finetune_para_type in [FineTuneParaType.NoFreeze, FineTuneParaType.FullFineTune]:
            self.optimizer = self.base_model_optimizer
        else:
            task_name = f"task{self.dataset.task_idx_list[self.train_on_ith_task]}"
            if self.train_on_ith_task == 0:
                if task_name in self.LoRA_opts.keys():
                    self.enable_lora_training(task_name=task_name)
                    self.optimizer = [self.base_model_optimizer, self.LoRA_opts[task_name]]
                else:
                    self.optimizer = self.entire_optimizer
            else:
                self.enable_lora_training(task_name=task_name)
                self.optimizer = self.LoRA_opts[task_name]

    def get_batch_data(self):
        data_from_task = self.dataset.task_idx_list[self.train_on_ith_task]
        if self.argus.finetune_para_type == FineTuneParaType.FullFineTune:
            rehearsal_sample = False
            if self.step % self.argus.rehearsal_frequency == 0 and self.train_on_ith_task > 0.5:
                rehearsal_sample = True
            if rehearsal_sample:
                if self.argus.partial_rehearsal:
                    batch, data_from_task = self.dataset.get_rehearsal_buffer_from_specific_task(
                        batch_size=self.argus.dm_batch_size, task_name=self.dataset.task_labels[random.choice(range(self.train_on_ith_task))], sample_bound=self.argus.rehearsal_sample_bound)
                else:
                    batch, data_from_task = self.dataset.get_rehearsal_buffer_from_all_previous_tasks(batch_size=self.argus.dm_batch_size, sample_bound=self.argus.rehearsal_sample_bound)
            else:
                batch = next(self.dataloader)
        elif self.argus.finetune_para_type in [FineTuneParaType.InMidOutConv1d, FineTuneParaType.CrossAttention]:
            batch = next(self.dataloader)
        elif self.argus.finetune_para_type == FineTuneParaType.NoFreeze:
            batch, data_from_task = self.dataset.get_rehearsal_buffer_from_specific_task(
                batch_size=self.argus.dm_batch_size, task_name=random.choice(self.dataset.task_names), sample_bound=self.argus.rehearsal_sample_bound)
        else:
            raise NotImplementedError
        batch = batch_to_device(batch, device=self.device, convert_to_torch_float=True)
        return batch, data_from_task

    def train(self):
        # self.step += 1
        train_start_time = time.time()
        for task_i, task_idx in enumerate(self.dataset.task_idx_list):
            self.train_on_ith_task = task_i
            self.dataset.set_task_idx(task_idx)
            self.dataloader = cycle_dataloader(argus=self.argus, dataset=self.dataset, train_batch_size=self.argus.dm_batch_size)
            self.switch_optimizer()
            for epoch_i in range(self.argus.n_epochs):
                for step in range(self.argus.steps_per_epoch):
                    if self.step % 10000 == 0:
                        self.save_training_hyperparameters()
                        self.save_parameter_flag = True
                    batch, data_from_task = self.get_batch_data()
                    loss, infos = self.model.loss(
                        x=batch.trajectories, cond=batch.conditions, task_identity=batch.task_identity, task_name=[self.dataset.task_labels[self.dataset.current_task_idx]], returns=batch.returns)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    if (self.step + 1) % self.gradient_accumulate_every == 0:
                        if isinstance(self.optimizer, list):
                            for opt_i in range(len(self.optimizer)):
                                self.optimizer[opt_i].step()
                                self.optimizer[opt_i].zero_grad()
                        else:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                    if self.wandb_log:
                        if self.step % self.wandb_log_frequency == 0:
                            wandb_infos = {}
                            wandb_infos.update(infos)
                            for info_key, info_val in wandb_infos.items():
                                wandb_infos[info_key] = info_val
                            wandb.log(wandb_infos, step=self.step)

                    if self.step % self.update_ema_every == 0:
                        self.step_ema()

                    if self.step % self.save_freq == 0 or (self.step % (self.argus.n_epochs * self.argus.steps_per_epoch) == (self.argus.n_epochs*self.argus.steps_per_epoch - 1)):
                        self.save_checpoint()
                        eval_results = self.multi_task_eval()
                        if self.wandb_log:
                            wandb.log(eval_results, step=self.step)
                    if self.step % self.log_freq == 0:
                        infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                        logger.print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {time.time()-train_start_time:8.4f} | current task {self.dataset.current_task_idx} | data from {data_from_task}')
                        metrics = {k:v for k, v in infos.items()}
                        metrics['steps'] = self.step
                        metrics['loss'] = loss.detach().item()
                        metrics['ProcessID'] = os.getpid()
                        logger.log_metrics_summary(metrics, default_stats='mean')
                        train_start_time = time.time()

                    self.step += 1

    def regular_loss(self):
        assert self.train_on_ith_task > 0.5
        regular_loss = 0
        if self.argus.finetune_para_type in [FineTuneParaType.ResidualTemporalBlock_TimeMlp, FineTuneParaType.TemporalDowns]:
            lora_parameters_list = []
            lora_parameters = {}
            for ith_task in range(self.train_on_ith_task + 1):
                if ith_task == self.train_on_ith_task:
                    lora_parameters_list.append(self.model.model.generate_LoRA_parameters(argus=self.argus, list_out=False))
                else:
                    saved_parameters, _, __ = self.read_previous_tasks_LoRA_parameters(task_idx=self.dataset.task_idx_list[0])
                    lora_parameters_list.append(saved_parameters)
            for ith_task in range(self.train_on_ith_task):
                if ith_task < self.train_on_ith_task - 1:
                    for key, val in lora_parameters_list[0].items():
                        if key not in lora_parameters.keys():
                            lora_parameters[key] = 0
                        else:
                            lora_parameters[key] += lora_parameters_list[ith_task+1][key].data.clone() - lora_parameters_list[ith_task][key].data.clone()
                else:
                    for key, val in lora_parameters_list[0].items():
                        if key not in lora_parameters.keys():
                            pairwise_product = lora_parameters_list[ith_task+1][key] - lora_parameters_list[ith_task][key].data.clone()
                        else:
                            pairwise_product = lora_parameters[key] * (lora_parameters_list[ith_task + 1][key] - lora_parameters_list[ith_task][key].data.clone())
                        regular_loss += torch.norm(pairwise_product, p=2)
        elif self.argus.finetune_para_type in [FineTuneParaType.CrossAttention]:
            lora_parameters = self.model.model.generate_LoRA_parameters(argus=self.argus, list_out=False)
            for key, val in lora_parameters.items():
                regular_loss += torch.norm(val, p=2)
        else:
            raise NotImplementedError
        return regular_loss

    def multi_task_eval(self):
        eval_results, total_mean_success_rate, total_mean_ruturn = {}, [], []
        for task_i, eval_task_idx in enumerate(self.dataset.task_idx_list):
            if self.dataset.env_name in ["cheetah_vel", "ant_dir", "cheetah_dir", "ML1-pick-place-v2"]:
                eval_result = multi_task_parallel_ant_dir_eval(
                    argus=self.argus, dataset=self.dataset, model=self.ema_model, eval_task_idx=eval_task_idx,
                    eval_episodes=self.argus.eval_episodes, ddim_sample=True, fix_normalizer_id=self.argus.fix_normalizer_id)
                for key, val in eval_result.items():
                    if "ave_return" in key:
                        total_mean_ruturn.append(val)
            elif self.dataset.env_name == "continual_world":
                eval_result = multi_task_parallel_continual_world_eval(
                    argus=self.argus, dataset=self.dataset, model=self.ema_model, eval_task_idx=eval_task_idx,
                    eval_episodes=self.argus.eval_episodes, ddim_sample=True, fix_normalizer_id=self.argus.fix_normalizer_id)
                for key, val in eval_result.items():
                    if "success_rate" in key:
                        total_mean_success_rate.append(val)
                    if "ave_return" in key:
                        total_mean_ruturn.append(val)
            else:
                raise NotImplementedError
            for result_key, result_val in eval_result.items():
                if result_key in list(eval_results.keys()):
                    result_key = f"cw20_{result_key}"
                eval_results[result_key] = result_val
            # eval_results.update(eval_result)
        if self.dataset.env_name in ["cheetah_vel", "ant_dir", "cheetah_dir", "ML1-pick-place-v2"]:
            eval_results.update({"total_mean_return": np.mean(total_mean_ruturn)})
        elif self.dataset.env_name == "continual_world":
            eval_results.update({"total_mean_success_rate": np.mean(total_mean_success_rate)})
            eval_results.update({"total_mean_return": np.mean(total_mean_ruturn)})
        else:
            raise NotImplementedError
        print(eval_results)
        return eval_results

    def save_training_hyperparameters(self):
        savepath = os.path.join(self.get_detailed_save_path(), 'hyperpara_config')
        os.makedirs(savepath, exist_ok=True)
        print(colored(f"Making {savepath}"))
        with open(os.path.join(savepath, "config.pickle"), "wb") as file:
            pickle.dump(obj2dict(self.argus), file)
        print(colored(f"Saving training_hyperparameters to {savepath}"))

    def save_checpoint(self):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            # 'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.get_detailed_save_path(), 'checkpoint')
        os.makedirs(savepath, exist_ok=True)
        # logger.save_torch(data, savepath)
        torch.save(data, os.path.join(savepath, f'state_{self.step}.pt'))
        torch.save(data, os.path.join(savepath, 'state.pt'))
        logger.print(f'[ utils/training ] Saved model to {savepath}')

    def save_LoRA_checpoint(self):
        savepath = os.path.join(self.get_detailed_save_path(), 'lora_checkpoint')
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        saved_parameters = self.model.model.generate_LoRA_parameters(argus=self.argus, list_out=False)
        torch.save(saved_parameters, os.path.join(savepath, f'{self.get_lora_model_prefix()}_task{self.dataset.current_task_idx}_{self.step}.pt'))
        logger.print(f'[ utils/lora_training ] Saved model to {savepath}')

    def get_all_checkpoints(self):
        checkpoints_list = os.listdir(os.path.join(self.get_detailed_save_path(), 'checkpoint'))
        checkpoints_step = []
        for checkpoint in checkpoints_list:
            if "_" in checkpoint:
                checkpoints_step.append(int(checkpoint.split("_")[-1].split(".")[0]))
        checkpoints_step.sort()
        return checkpoints_step

    def get_checkpoint_step(self, loadpath, step_offset, checkpoint_identify=None):
        checkpoints_list = os.listdir(loadpath)
        checkpoints_step = []
        for checkpoint in checkpoints_list:
            if checkpoint_identify is not None:
                if checkpoint_identify in checkpoint:
                    checkpoints_step.append(int(checkpoint.split("_")[-1].split(".")[0]))
            else:
                if "_" in checkpoint:
                    checkpoints_step.append(int(checkpoint.split("_")[-1].split(".")[0]))
        checkpoints_step.sort()
        if step_offset=="latest":
            return checkpoints_step[-1]
        else:
            return checkpoints_step[0] + step_offset

    def load(self, step_offset="latest"):
        '''
            loads model and ema from disk
        '''
        step = self.get_checkpoint_step(
            loadpath=os.path.join(self.get_detailed_save_path(), 'checkpoint'), step_offset=step_offset)
        if step_offset != "latest":
            loadpath = os.path.join(self.get_detailed_save_path(), f'checkpoint/state_{step}.pt')
        else:
            loadpath = os.path.join(self.get_detailed_save_path(), f'checkpoint/state.pt')
        # data = logger.load_torch(loadpath)
        data = torch.load(loadpath)

        self.step = data['step']
        print(colored(f" From {loadpath} Load Model with saved train step {self.step} Success !!!", color="green"))
        self.model.load_state_dict(data['ema'])
        self.ema_model.load_state_dict(data['ema'])

    def read_previous_tasks_LoRA_parameters(self, task_idx, step_offset="latest"):
        savepath = os.path.join(self.get_detailed_save_path(), 'lora_checkpoint')
        step = self.get_checkpoint_step(
            loadpath=os.path.join(self.get_detailed_save_path(), 'lora_checkpoint'), step_offset=step_offset,
            checkpoint_identify=f"task{task_idx}")
        loadpath = os.path.join(savepath, f'{self.get_lora_model_prefix()}_task{task_idx}_{step}.pt')
        saved_parameters = torch.load(loadpath)
        return saved_parameters, loadpath, step

    def load_LoRA(self, task_idx, step_offset="latest"):
        saved_parameters, loadpath, step = self.read_previous_tasks_LoRA_parameters(task_idx=task_idx, step_offset=step_offset)
        self.model.model.load_LoRA_parameters(argus=self.argus, saved_parameters=saved_parameters)
        print(colored(f" From {loadpath} Load LoRA Model with saved train step {step} Success !!!", color="green"))

    def parameters_analysis(self):
        return self.model.parameters_analysis()


