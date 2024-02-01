import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import numpy as np
import fire
from trainer.trainer_util import seed_configuration
from config import *
from datasets.sequence_dataset import SequenceDataset
from models.diffusion_model import GaussianDiffusion
from models.temporal_unet import TemporalUnet
from trainer.diffuser_trainer import Trainer
import pickle

def modify_parameters_for_continual_training(hyperparameters):
    assert hyperparameters["task_idx"] == eval_training["task_idx"]
    hyperparameters[f"original_task_idx"] = eval_training["task_idx"]
    if "original_current_exp_label" in list(hyperparameters.keys()):
        hyperparameters[f"original_current_exp_label"] = eval_training[f"current_exp_label"]
    if "ddim_stride" in list(hyperparameters.keys()):
        hyperparameters[f"ddim_stride"] = eval_training[f"ddim_stride"]
    additional_hyperparameters, alternative_hyperparameters = {}, {}
    for key, val in eval_training.items():
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

def eval_current_saved_model(config_para_path, dataset_task_idx, eval_task_idx):
    with open(os.path.join(config_para_path, "config.pickle"), "rb") as file:
        hyperparameters = pickle.load(file)
    argus = dict2obj(modify_parameters_for_continual_training(hyperparameters=hyperparameters))
    # argus.wandb_log = False
    dataset = SequenceDataset(
        argus=argus, project_path=None, task_idx=dataset_task_idx, env_name=argus.dataset, sequence_length=argus.sequence_length,
        normalizer=argus.normalizer, termination_penalty=argus.termination_penalty,
        discount=argus.discount, returns_scale=argus.returns_scale,
    )
    argus.input_channels = dataset.observation_dim + dataset.action_dim
    argus.out_channels = dataset.observation_dim + dataset.action_dim
    argus.observation_dim = dataset.observation_dim
    argus.action_dim = dataset.action_dim
    argus.condition_dim = dataset.observation_dim
    temporal_unet = TemporalUnet(
        sequence_length=argus.sequence_length, input_channels=argus.input_channels, out_channels=argus.out_channels,
        dim=argus.dim, dim_mults=argus.dim_mults)
    diffusion_model = GaussianDiffusion(
        model=temporal_unet, horizon=argus.sequence_length, observation_dim=argus.observation_dim, action_dim=argus.action_dim,
        condition_dim=argus.condition_dim, argus=argus)
    trainer = Trainer(diffusion_model=diffusion_model, dataset=dataset, argus=argus)
    for i in range(20):
        try:
            trainer.load(step_offset=50000*i)
            trainer.eval(eval_task_idx=eval_task_idx)
        except:
            print(f"Eval error: can not find model at step_offset {50000*i}")
            break