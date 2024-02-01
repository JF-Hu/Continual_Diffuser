import torch
import enum

class DiffusionConditionType(enum.Enum):
    unconditional = enum.auto()
    classifier_free = enum.auto()

class UnetType(enum.Enum):
    temporal_unet = enum.auto()
    spatial_transformer_unet = enum.auto()
    return_based_temporal_unet = enum.auto()
    return_based_inverse_dynamics_temporal_unet = enum.auto()

class FineTuneParaType(enum.Enum):
    NoFreeze = enum.auto()
    FullFineTune = enum.auto()
    ResidualTemporalBlock_TimeMlp = enum.auto()
    TemporalDowns = enum.auto()
    InMidOutConv1d = enum.auto()
    CrossAttention = enum.auto()

base_parameters = dict(

    # todo    dataset setting
    dataset='None',
    termination_penalty=-100,
    discount=0.99,
    returns_scale=400,
    normalizer='GaussianNormalizer',   # CDFNormalizer GaussianNormalizer  MinMaxNormalizer

    # todo    training setting
    loss_type = 'l2',
    loss_discount = 1,
    loss_weights = None,
    ema_decay = 0.995,
    batch_size = 32,
    learning_rate = 2e-4,
    gradient_accumulate_every = 2,
    log_freq = 1000,
    sample_freq = 10000,
    save_freq = 10000,
    save_parallel = False,
    save_checkpoints = False,
    wandb_log = False,
    wandb_exp_name = None,
    wandb_exp_group = None,
    wandb_log_frequency = 2000,
    wandb_project_name = 'diffusion_offline',
    save_range = [100000, 900000],
    n_train_steps = 1000000,
    n_steps_per_epoch = 10000,
    save_path = './output',
    current_exp_label = "DD_baseline",


    # todo    inverse dtnamics
    ar_inv = False,
    train_only_inv = False,

    # todo    diffusion model setting
    sequence_length = 100,
    diffusion_steps = 200,
    clip_denoised = True,
    predict_epsilon = True,
    hidden_dim = 256,
    action_weight = 10,
    returns_condition = True,
    condition_guidance_w = 1.2,
    goals_condition = False,
    condition_dropout = 0.25,
    calc_energy = False,
    dim_mults = (1, 4, 8),
    dim = 128,

    # todo    device setting
    device = "cuda" if torch.cuda.is_available() else "cpu",

    # todo    seed setting
    reset_seed = True,
    seed = 100,

    # todo    evaluation setting
    num_eval = 30,

    ddim_stride = 10,
)

continual_diffuser = dict(
    dataset='ant_dir',
    ddim_stride=20,
    task_idx=10,
    normalizer='GaussianNormalizer',
    train_with_normed_data=True,
    current_exp_label = "test",
    input_channels = None,
    out_channels = None,
    task_identity_dim = None,
    action_ranges = None,

    dataset_resample = False,
    termination_penalty = 0,
    n_epochs = 50,
    steps_per_epoch = 10000,
    sequence_length = 48,
    eval_episodes = 30,

    dm_update_ema_every = 10,
    dm_step_start_ema = 2000,
    dm_log_freq = 1000,
    dm_save_freq = 50000,
    dm_batch_size = 32,
    dm_gradient_accumulate_every = 2,
    dm_lr = 3e-4,

    wandb_log = False,
    task_idx_list = ["10"],
    continual_world_dataset_quality = "expert",
    unet_type = UnetType.temporal_unet,
    finetune_para_type = FineTuneParaType.InMidOutConv1d,
    condition_type = DiffusionConditionType.classifier_free,
    regularization_coef = 0.01,
    rehearsal_frequency = 14,
    rehearsal_sample_bound = 0.01,
    partial_rehearsal = True,
    fix_normalizer_id = False,
    lora_dim = 64,
    continual_world_data_type = "pkl",

    debug_mode = False,
    dataset_load_min = 2000,
    dataset_load_max = 4000,

    offline_cl_baseline_name = "ewc"
)

eval_training = dict(
    dataset='ant_dir',
    task_idx=10,
    ddim_stride = 10,
    current_exp_label = "continual_train_gaus_norm",
)

continual_training = dict(
    dataset='ant_dir',
    task_idx=21,
    ddim_stride = 10,
    current_exp_label = "continual_train",
)
















