import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.model_util import cosine_beta_schedule, extract, replaceable_condition, Losses
from models.sampler import DDIMSampler
from models.temporal_unet import TemporalUnet
from models.task_cond_temporal_unet import TaskBasedTemporalUnet
from models.lora_task_cond_temporal_unet import LoRATaskBasedTemporalUnet
from config.hyperparameter import UnetType, FineTuneParaType, DiffusionConditionType

def _to_str(num):
	if num >= 1e6:
		return f'{(num/1e6):.2f} M'
	else:
		return f'{(num/1e3):.2f} k'

def param_to_module(param):
	module_name = param[::-1].split('.', maxsplit=1)[-1][::-1]
	return module_name

class TaskCondInverseDynamicGaussianDiffusion(nn.Module):

    def __init__(
            self, model, horizon, observation_dim, action_dim, condition_dim, argus
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.condition_dim = argus.condition_dim
        self.argus = argus
        self.stride = argus.ddim_stride
        self.condition_guidance_w = argus.condition_guidance_w
        betas = cosine_beta_schedule(argus.diffusion_steps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        self.n_timesteps = int(argus.diffusion_steps)
        self.clip_denoised = argus.clip_denoised
        self.predict_epsilon = argus.predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses['NormalL2']()
        self.ddim_sampler = DDIMSampler(alphas=alphas, alphas_cumprod=alphas_cumprod, stride=argus.ddim_stride, eta=1)

        self.inverse_dynamic_model = nn.Sequential(
            nn.Linear(2 * self.observation_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
        )

    #------------------------------------------ sampling ------------------------------------------#
    def apply_condition(self, x, cond):
        x = replaceable_condition(x=x, conditions=cond, start_dim=0, end_dim=self.observation_dim)
        return x

    def x_clip_opt(self, x, action_range=(-0.99999, 0.99999)):
        if self.clip_denoised:
            x = x.clamp(np.min(action_range), np.max(action_range))
        else:
            assert RuntimeError()
        return x

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # @torch.no_grad()
    def p_mean_variance(self, x, cond, t, task_identity, task_name, action_range):
        if self.argus.condition_type == DiffusionConditionType.classifier_free:
            epsilon_cond = self.model(x=x, cond=cond, timesteps=t, task_identity=task_identity, context=task_identity, task_name=task_name, use_dropout=False)
            epsilon_uncond = self.model(x=x, cond=cond, timesteps=t, task_identity=task_identity, context=task_identity, task_name=task_name, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w * (epsilon_cond - epsilon_uncond)
        elif self.argus.condition_type == DiffusionConditionType.unconditional:
            epsilon = self.model(x=x, cond=cond, timesteps=t, task_identity=task_identity, context=task_identity, task_name=task_name, use_dropout=False)
        else:
            raise NotImplementedError
        # if self.argus.unet_type in [UnetType.temporal_unet]:
        #     epsilon_cond = self.model(x=x, cond=cond, timesteps=t, task_identity=task_identity, task_name=task_name, context=None, use_dropout=False)
        #     epsilon_uncond = self.model(x=x, cond=cond, timesteps=t, task_identity=task_identity, task_name=task_name, context=None, force_dropout=True)
        # elif self.argus.unet_type in [UnetType.spatial_transformer_unet]:
        #     epsilon_cond = self.model(x=x, cond=cond, timesteps=t, context=task_identity, task_name=task_name, use_dropout=False)
        #     epsilon_uncond = self.model(x=x, cond=cond, timesteps=t, context=task_identity, task_name=task_name, force_dropout=True)
        # else:
        #     raise NotImplementedError
        # epsilon = epsilon_uncond + self.condition_guidance_w * (epsilon_cond - epsilon_uncond)
        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)
        x_recon = self.x_clip_opt(x=x_recon, action_range=action_range)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    @torch.no_grad()
    def p_sample(self, x, cond, t, task_identity, task_name, action_range):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_recon = self.p_mean_variance(x=x, cond=cond, t=t, task_identity=task_identity, task_name=task_name, action_range=action_range)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        x_t_1 = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return x_t_1

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, task_identity, task_name, return_diffusion=False, ddim_sample=False, action_range=(-0.99999, 0.99999), *args, **kwargs):
        device = self.betas.device
        x = torch.randn(shape, device=device)
        x = self.apply_condition(x=x, cond=cond)
        if return_diffusion: diffusion = [x]
        if ddim_sample:
            for i in reversed(range(0, self.ddim_sampler.generation_step)):
                actual_t = torch.full((shape[0],), i * self.stride, device=device, dtype=torch.long)
                if self.argus.condition_type == DiffusionConditionType.classifier_free:
                    epsilon_cond = self.model(x=x, cond=cond, timesteps=actual_t, task_identity=task_identity, context=task_identity, task_name=task_name, use_dropout=False)
                    epsilon_uncond = self.model(x=x, cond=cond, timesteps=actual_t, task_identity=task_identity, context=task_identity, task_name=task_name, force_dropout=True)
                    epsilon = epsilon_uncond + self.condition_guidance_w * (epsilon_cond - epsilon_uncond)
                elif self.argus.condition_type == DiffusionConditionType.unconditional:
                    epsilon = self.model(x=x, cond=cond, timesteps=actual_t, task_identity=task_identity, context=task_identity, task_name=task_name, use_dropout=False)
                else:
                    raise NotImplementedError
                # if self.argus.unet_type in [UnetType.temporal_unet]:
                #     if self.argus.condition_type == DiffusionConditionType.classifier_free:
                #         epsilon_cond = self.model(x=x, cond=cond, timesteps=actual_t, task_identity=task_identity, task_name=task_name, context=None, use_dropout=False)
                #         epsilon_uncond = self.model(x=x, cond=cond, timesteps=actual_t, task_identity=task_identity, task_name=task_name, context=None, force_dropout=True)
                #         epsilon = epsilon_uncond + self.condition_guidance_w * (epsilon_cond - epsilon_uncond)
                #     elif self.argus.condition_type == DiffusionConditionType.unconditional:
                #         epsilon_cond = self.model(x=x, cond=cond, timesteps=actual_t, task_identity=task_identity, task_name=task_name, context=None, use_dropout=False)
                # elif self.argus.unet_type in [UnetType.spatial_transformer_unet]:
                #     epsilon_cond = self.model(x=x, cond=cond, timesteps=actual_t, context=task_identity, task_name=task_name, use_dropout=False)
                #     epsilon_uncond = self.model(x=x, cond=cond, timesteps=actual_t, context=task_identity, task_name=task_name, force_dropout=True)
                #     epsilon = epsilon_uncond + self.condition_guidance_w * (epsilon_cond - epsilon_uncond)
                # else:
                #     raise NotImplementedError
                x = self.ddim_sampler.sample_one_step(epsilon=epsilon, x_t=x, t=i)
                x = self.x_clip_opt(x=x, action_range=action_range)
                x = self.apply_condition(x=x, cond=cond)
                if return_diffusion: diffusion.append(x)
        else:
            # progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
            for i in reversed(range(0, self.n_timesteps)):
                timesteps = torch.full((shape[0],), i, device=device, dtype=torch.long)
                x = self.p_sample(x=x, cond=cond, t=timesteps, task_identity=task_identity, task_name=task_name, action_range=action_range)
                x = self.apply_condition(x=x, cond=cond)
                if return_diffusion: diffusion.append(x)
        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, data_shape, task_identity, ddim_sample, *args, **kwargs):
        self.model.eval()
        self.inverse_dynamic_model.eval()
        return self.p_sample_loop(shape=data_shape, cond=cond, task_identity=task_identity, ddim_sample=ddim_sample, *args, **kwargs)
    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, task_identity, task_name, returns, *args, **kwargs):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = self.apply_condition(x=x_noisy, cond=cond)
        if self.argus.unet_type in [UnetType.temporal_unet, UnetType.return_based_temporal_unet, UnetType.return_based_inverse_dynamics_temporal_unet]:
            x_recon = self.model(x=x_noisy, cond=cond, timesteps=t, task_identity=task_identity, task_name=task_name, returns=returns)
        elif self.argus.unet_type in [UnetType.spatial_transformer_unet]:
            x_recon = self.model(x=x_noisy, cond=cond, timesteps=t, context=task_identity, task_name=task_name, returns=returns)
        else:
            raise NotImplementedError
        x_recon = self.apply_condition(x=x_recon, cond=cond)
        assert noise.shape == x_recon.shape
        if self.predict_epsilon:
            # noise = self.apply_condition(x=noise, cond=cond)
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)
        return loss, info

    def inverse_dynamic_loss(self, x):
        x_t = x[:, :-1, :self.observation_dim]
        a_t = x[:, :-1, self.observation_dim:]
        x_t_1 = x[:, 1:, :self.observation_dim]
        x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
        x_comb_t = x_comb_t.reshape(-1, 2 * self.observation_dim)
        a_t = a_t.reshape(-1, self.action_dim)
        pred_a_t = self.inverse_dynamic_model(x_comb_t)
        loss = F.mse_loss(pred_a_t, a_t)
        return loss

    def loss(self, x, cond, task_identity, task_name, returns, *args, **kwargs):
        self.model.train()
        self.inverse_dynamic_model.train()
        inv_loss = self.inverse_dynamic_loss(x)
        info = {}
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        diffuse_loss, diffuse_loss_info = self.p_losses(x[:, :, :self.observation_dim], cond, t, task_identity, task_name, returns, *args, **kwargs)
        total_loss = diffuse_loss + inv_loss
        info.update({"diffusion_loss": diffuse_loss.detach().cpu().numpy(), "inv_loss": inv_loss.detach().cpu().numpy()})
        return total_loss, info

    def forward(self, cond, data_shape, task_identity, ddim_sample, *args, **kwargs):
        return self.conditional_sample(cond=cond, data_shape=data_shape, task_identity=task_identity, ddim_sample=ddim_sample, *args, **kwargs)

    def get_action(self, cond, data_shape, task_identity, ddim_sample, task_name, *args, **kwargs):
        state_seq = self.conditional_sample(cond, data_shape, task_identity, ddim_sample, task_name=task_name, *args, **kwargs)
        x_comb_t = torch.cat([state_seq[:, :-1, :], state_seq[:, 1:, :]], dim=-1)
        actions = self.inverse_dynamic_model(x_comb_t)
        return actions

    def generate_inverse_dynamics_parameters(self, argus, list_out=True):
        inv_parameters = [] if list_out else {}
        for name, para in self.inverse_dynamic_model.named_parameters():
            if "lora_A" not in name and "lora_B" not in name:
                if list_out:
                    inv_parameters.append(para)
                else:
                    inv_parameters.update({name: para})
        return inv_parameters

    def generate_base_model_parameters(self, argus, list_out=True):
        total_parameter = [] if list_out else {}
        unet_parameters = self.model.generate_base_model_parameters(argus=argus, list_out=list_out)
        inverse_dynamic_parameters = self.generate_inverse_dynamics_parameters(argus=argus, list_out=list_out)
        if list_out:
            total_parameter.extend(unet_parameters)
            total_parameter.extend(inverse_dynamic_parameters)
        else:
            total_parameter.update(unet_parameters)
            total_parameter.update(inverse_dynamic_parameters)
        return total_parameter

    def report_parameters(self, model, topk=10):
        counts = {k: p.numel() for k, p in model.named_parameters()}
        n_parameters = sum(counts.values())
        print(f'[ utils/arrays ] Total parameters: {_to_str(n_parameters)}')

        modules = dict(model.named_modules())
        sorted_keys = sorted(counts, key=lambda x: -counts[x])
        max_length = max([len(k) for k in sorted_keys])
        if len(sorted_keys) < topk:
            topk = len(sorted_keys)
        for i in range(topk):
            key = sorted_keys[i]
            count = counts[key]
            module = param_to_module(key)
            print(' ' * 8, f'{key:10}: {_to_str(count)} | {modules[module]}')

        remaining_parameters = sum([counts[k] for k in sorted_keys[topk:]])
        print(' ' * 8, f'... and {len(counts) - topk} others accounting for {_to_str(remaining_parameters)} parameters')
        return n_parameters

    def parameters_analysis(self):
        results = {}
        if isinstance(self.model, (TemporalUnet, TaskBasedTemporalUnet, LoRATaskBasedTemporalUnet)):
            for param_name, param in self.model.named_parameters():
                results.update({f"{param_name}.mean": param.data.mean().detach().cpu().numpy().item(), f"{param_name}.var": param.data.var().detach().cpu().numpy().item()})
        else:
            raise NotImplementedError
        return results