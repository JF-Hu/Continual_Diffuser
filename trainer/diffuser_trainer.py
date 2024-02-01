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
from evaluation.parallel_evaluation import parallel_ant_dir_eval


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

class Trainer(object):
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
        self.dataloader = cycle_dataloader(argus=self.argus, dataset=self.dataset, train_batch_size=argus.dm_batch_size)
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
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=argus.dm_lr)
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
        # self.save_training_parameters()
        if self.wandb_log:
            wandb.init(name=self.wandb_exp_name, group=self.wandb_exp_group, project=self.wandb_project_name, config=obj2dict(self.argus))

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def train(self):
        train_start_time = time.time()
        for epoch_i in range(self.argus.n_epochs):
            for step in range(self.argus.steps_per_epoch):
                if epoch_i % 20 == 0:
                    self.save_training_hyperparameters()
                    self.save_parameter_flag = True
                for i in range(self.gradient_accumulate_every):
                    batch = next(self.dataloader)
                    batch = batch_to_device(batch, device=self.device, convert_to_torch_float=True)
                    loss, infos = self.model.loss(*batch)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.wandb_log:
                    if self.step % self.wandb_log_frequency == 0:
                        wandb_infos = {}
                        wandb_infos.update(infos)
                        # wandb_infos.update({"diffusion_loss": loss})
                        for info_key, info_val in wandb_infos.items():
                            wandb_infos[info_key] = info_val
                        wandb.log(wandb_infos, step=self.step)

                if self.step % self.update_ema_every == 0:
                    self.step_ema()

                if self.step % self.save_freq == 0:
                    self.save_checpoint()

                if self.step % self.log_freq == 0:
                    infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                    logger.print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {time.time()-train_start_time:8.4f}')
                    metrics = {k:v for k, v in infos.items()}
                    metrics['steps'] = self.step
                    metrics['loss'] = loss.detach().item()
                    metrics['ProcessID'] = os.getpid()
                    logger.log_metrics_summary(metrics, default_stats='mean')
                    train_start_time = time.time()
                self.step += 1

    def eval(self, eval_task_idx):
        eval_results = parallel_ant_dir_eval(
            argus=self.argus, dataset=self.dataset, model=self.ema_model, eval_task_idx=eval_task_idx, eval_episodes=30, ddim_sample=True)
        if self.wandb_log:
            wandb.log(eval_results, step=self.step)

    def save_training_hyperparameters(self):
        savepath = os.path.join(self.save_path, "pretrained_model", "_".join(self.env_name.split("-")), f"task{self.argus.task_idx}", self.current_exp_label, 'hyperpara_config')
        os.makedirs(savepath, exist_ok=True)
        with open(os.path.join(savepath, "config.pickle"), "wb") as file:
            pickle.dump(obj2dict(self.argus), file)

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
        savepath = os.path.join(self.save_path, "pretrained_model", "_".join(self.env_name.split("-")), f"task{self.argus.task_idx}", self.current_exp_label, 'checkpoint')
        os.makedirs(savepath, exist_ok=True)
        # logger.save_torch(data, savepath)
        torch.save(data, os.path.join(savepath, f'state_{self.step}.pt'))
        torch.save(data, os.path.join(savepath, 'state.pt'))
        logger.print(f'[ utils/training ] Saved model to {savepath}')

    def get_checkpoint_step(self, loadpath, step_offset):
        checkpoints_list = os.listdir(loadpath)
        checkpoints_step = []
        for checkpoint in checkpoints_list:
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
            loadpath=os.path.join(self.save_path, "pretrained_model", "_".join(self.env_name.split("-")), f"task{self.argus.original_task_idx}", self.argus.original_current_exp_label, 'checkpoint'), step_offset=step_offset)
        if step_offset != "latest":
            loadpath = os.path.join(self.save_path, "pretrained_model", "_".join(self.env_name.split("-")), f"task{self.argus.original_task_idx}", self.argus.original_current_exp_label, f'checkpoint/state_{step}.pt')
        else:
            loadpath = os.path.join(self.save_path, "pretrained_model", "_".join(self.env_name.split("-")), f"task{self.argus.original_task_idx}", self.argus.original_current_exp_label, f'checkpoint/state.pt')
        # data = logger.load_torch(loadpath)
        data = torch.load(loadpath)

        self.step = data['step']
        print(colored(f" From {loadpath} Load Model with saved train step {self.step} Success !!!", color="green"))
        self.model.load_state_dict(data['ema'])
        self.ema_model.load_state_dict(data['ema'])