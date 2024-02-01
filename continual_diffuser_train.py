import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import numpy as np
import fire
from trainer.trainer_util import seed_configuration
from config import *
from datasets.sequence_dataset import MultiTaskSequenceDataset
from models.task_cond_diffusion_model import TaskCondGaussianDiffusion
from models.task_cond_inversedyn_diffusion_model import TaskCondInverseDynamicGaussianDiffusion
from models.task_return_cond_inversedyn_diffusion_model import TaskReturnCondInverseDynamicGaussianDiffusion
from models.task_return_cond_diffusion_model import TaskReturnCondGaussianDiffusion
from models.task_cond_temporal_unet import  TaskBasedTemporalUnet
from models.task_return_cond_temporal_unet import TaskReturnBasedTemporalUnet
from models.transformer_based_unet import SpatialTransformerUnet
from models.lora_transformer_based_unet import SpatialLoRATransformerUnet
from models.lora_task_cond_temporal_unet import LoRATaskBasedTemporalUnet
from models.lora_temporal_unet import LoRATemporalUnet
from trainer.task_cond_diffuser_trainer import TaskCondTrainer
from trainer.trainer_util import batchify
from termcolor import colored
import pickle
from config.hyperparameter import UnetType, FineTuneParaType
from evaluation.eval_current_model import eval_current_saved_model
from datasets.dataset_util import get_multi_task_name


def debug_mode_setting(argus):
    if argus.debug_mode:
        argus.eval_episodes = 2
        argus.dataset_load_min = 1500
        argus.dataset_load_max = 1600
        argus.wandb_log = False
    return argus

def generate_wandb_exp_name(argus, wandb_project='continual_diffuser'):
    multi_task_name = get_multi_task_name(argus=argus)
    argus.wandb_exp_name = f'{argus.dataset}-{multi_task_name}-{argus.current_exp_label}'
    if len(argus.dataset.split("-")) > 2:
        group_name = "-".join(argus.dataset.split("-")[0:2])
    else:
        group_name = argus.dataset
    argus.wandb_exp_group = group_name
    argus.wandb_project_name = wandb_project
    return argus

def train_test(argus, dataset, diffuser):
    print(colored("Model Train Testing ......", color="green"))
    batch = batchify(dataset[0], argus.device)
    loss, _ = diffuser.loss(*batch)
    loss.backward()
    print(colored("Success !!!", color="green"))

def import_parameters(var_kwargs=None, mode="normal"):
    print(colored(f"Loading {mode} hyperparameters ......", color="green"))
    if mode == "base":
        hyperparameters = base_parameters
        hyperparameters["mode"] = "normal"
    elif mode == "continual_diffuser":
        hyperparameters = continual_diffuser
        hyperparameters["mode"] = "pretrain"
        for key, val in base_parameters.items():
            if key not in hyperparameters.keys():
                hyperparameters[key] = val
        if var_kwargs:
            hyperparameters.update(var_kwargs)
    else:
        raise Exception("The mode of import_parameters is wrong !!!")
    return hyperparameters

def modify_parameters_for_continual_training(hyperparameters):
    assert hyperparameters["task_idx"] != continual_training["task_idx"]
    if "original_current_exp_label" in list(hyperparameters.keys()):
        hyperparameters[f"original_current_exp_label"] = continual_training[f"current_exp_label"]
    additional_hyperparameters, alternative_hyperparameters = {}, {}
    for key, val in continual_training.items():
        if key not in hyperparameters.keys():
            additional_hyperparameters[key] = val
            hyperparameters[key] = val
        else:
            if val != hyperparameters[key]:
                alternative_hyperparameters[key] = f"old[{hyperparameters[key]}]->new[{val}]"
                hyperparameters[f"original_{key}"] = hyperparameters[key]
                hyperparameters[key] = val
    print(f"additional hyperparameters: {additional_hyperparameters}\nalternative_hyperparameters:{alternative_hyperparameters}")
    return hyperparameters

def initial_train(**var_kwargs):
    argus = dict2obj(import_parameters(var_kwargs, mode="continual_diffuser"))
    argus = generate_wandb_exp_name(argus=argus, wandb_project="continual_diffuser")
    argus = seed_configuration(argus=argus)
    argus = debug_mode_setting(argus)
    print(obj2dict(argus))
    dataset = MultiTaskSequenceDataset(
        task_idx_list=argus.task_idx_list, argus=argus, project_path=None, env_name=argus.dataset, sequence_length=argus.sequence_length,
        normalizer=argus.normalizer, termination_penalty=argus.termination_penalty,
        discount=argus.discount, returns_scale=argus.returns_scale,
    )
    if argus.unet_type == UnetType.return_based_inverse_dynamics_temporal_unet:
        argus.input_channels = dataset.observation_dim
        argus.out_channels = dataset.observation_dim
    else:
        argus.input_channels = dataset.observation_dim + dataset.action_dim
        argus.out_channels = dataset.observation_dim + dataset.action_dim
    argus.observation_dim = dataset.observation_dim
    argus.action_dim = dataset.action_dim
    argus.condition_dim = dataset.observation_dim
    argus.task_identity_dim = dataset.task_identity_dim
    argus.action_ranges = dataset.action_ranges
    if argus.unet_type in [UnetType.spatial_transformer_unet]:
        if argus.finetune_para_type in [FineTuneParaType.CrossAttention]:
            unet = SpatialLoRATransformerUnet(
                input_channels=argus.input_channels, sequence_length=argus.sequence_length,
                out_channels=argus.out_channels,
                num_res_blocks=1, attention_resolutions=(4, 2, 1), channel_mult=(1, 2, 4), num_heads=8,
                model_channels=64, task_names=[f"task{task_i}" for task_i in dataset.task_idx_list], lora_dim=64, context_dim=1, transformer_depth=1, same_conv_kernel_size=3
            )
        else:
            unet = SpatialTransformerUnet(
                input_channels=argus.input_channels, sequence_length=argus.sequence_length, out_channels=argus.out_channels,
                num_res_blocks=1, attention_resolutions=(4,2,1), channel_mult=(1, 2, 4), num_heads=8,
                model_channels=64, context_dim=1, transformer_depth=1, same_conv_kernel_size=3
            )
    elif argus.unet_type in [UnetType.temporal_unet]:
        if argus.finetune_para_type in [FineTuneParaType.InMidOutConv1d]:
            unet = LoRATaskBasedTemporalUnet(
                sequence_length=argus.sequence_length, input_channels=argus.input_channels, out_channels=argus.out_channels, dim=argus.dim, dim_mults=argus.dim_mults,
                task_names=[f"task{task_i}" for task_i in dataset.task_idx_list], lora_dim=argus.lora_dim)
        else:
            unet = TaskBasedTemporalUnet(
                sequence_length=argus.sequence_length, input_channels=argus.input_channels, out_channels=argus.out_channels, task_identity_dim=argus.task_identity_dim,
                dim=argus.dim, dim_mults=argus.dim_mults)
    elif argus.unet_type in [UnetType.return_based_temporal_unet, UnetType.return_based_inverse_dynamics_temporal_unet]:
        if argus.finetune_para_type in [FineTuneParaType.FullFineTune]:
            unet = TaskReturnBasedTemporalUnet(
                sequence_length=argus.sequence_length, input_channels=argus.input_channels, out_channels=argus.out_channels, task_identity_dim=argus.task_identity_dim,
                dim_mults=argus.dim_mults)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    if argus.unet_type in [UnetType.spatial_transformer_unet, UnetType.temporal_unet]:
        diffusion_model = TaskCondGaussianDiffusion(
            model=unet, horizon=argus.sequence_length, observation_dim=argus.observation_dim, action_dim=argus.action_dim, condition_dim=argus.condition_dim, argus=argus)
    elif argus.unet_type in [UnetType.return_based_temporal_unet]:
        diffusion_model = TaskReturnCondGaussianDiffusion(
            model=unet, horizon=argus.sequence_length, observation_dim=argus.observation_dim, action_dim=argus.action_dim, condition_dim=argus.condition_dim, argus=argus)
    elif argus.unet_type in [UnetType.return_based_inverse_dynamics_temporal_unet]:
        diffusion_model = TaskReturnCondInverseDynamicGaussianDiffusion(
            model=unet, horizon=argus.sequence_length, observation_dim=argus.observation_dim, action_dim=argus.action_dim, condition_dim=argus.condition_dim, argus=argus)
    else:
        raise NotImplementedError
    trainer = TaskCondTrainer(
        diffusion_model=diffusion_model, dataset=dataset, argus=argus)
    trainer.train()

if __name__ == "__main__":
    fire.Fire(initial_train)