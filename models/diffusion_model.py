import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.model_util import cosine_beta_schedule, extract, replaceable_condition, Losses
from models.sampler import DDIMSampler

class GaussianDiffusion(nn.Module):

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

    #------------------------------------------ sampling ------------------------------------------#
    def apply_condition(self, x, cond):
        x = replaceable_condition(x=x, conditions=cond, start_dim=0, end_dim=self.observation_dim)
        return x

    def x_clip_opt(self, x):
        if self.clip_denoised:
            x = x.clamp(-0.99999, 0.99999)
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
    def p_mean_variance(self, x, cond, t):
        epsilon = self.model(x, cond, t)
        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)
        x_recon = self.x_clip_opt(x=x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    @torch.no_grad()
    def p_sample(self, x, cond, t):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_recon = self.p_mean_variance(x=x, cond=cond, t=t)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        x_t_1 = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return x_t_1

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, return_diffusion=False, ddim_sample=False):
        device = self.betas.device
        x = torch.randn(shape, device=device)
        x = self.apply_condition(x=x, cond=cond)
        if return_diffusion: diffusion = [x]
        if ddim_sample:
            for i in reversed(range(0, self.ddim_sampler.generation_step)):
                actual_t = torch.full((shape[0],), i * self.stride, device=device, dtype=torch.long)
                epsilon = self.model(x=x, timesteps=actual_t, cond=cond)
                x = self.ddim_sampler.sample_one_step(epsilon=epsilon, x_t=x, t=i)
                x = self.x_clip_opt(x=x)
                x = self.apply_condition(x=x, cond=cond)
                if return_diffusion: diffusion.append(x)
        else:
            # progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
            for i in reversed(range(0, self.n_timesteps)):
                timesteps = torch.full((shape[0],), i, device=device, dtype=torch.long)
                x = self.p_sample(x=x, cond=cond, t=timesteps)
                x = self.apply_condition(x=x, cond=cond)
                if return_diffusion: diffusion.append(x)
        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, data_shape, ddim_sample, *args, **kwargs):
        self.model.eval()
        return self.p_sample_loop(shape=data_shape, cond=cond, ddim_sample=ddim_sample, *args, **kwargs)
    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = self.apply_condition(x=x_noisy, cond=cond)
        x_recon = self.model(x=x_noisy, cond=cond, timesteps=t)
        x_recon = self.apply_condition(x=x_recon, cond=cond)
        assert noise.shape == x_recon.shape
        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)
        return loss, info

    def loss(self, x, cond, returns=None, rewards=None, goals=None, *args, **kwargs):
        info = {}
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        diffuse_loss, diffuse_loss_info = self.p_losses(x, cond, t)
        info.update({"diffusion_loss": diffuse_loss.detach().cpu().numpy()})
        return diffuse_loss, diffuse_loss_info

    def forward(self, cond, data_shape, ddim_sample, *args, **kwargs):
        return self.conditional_sample(cond=cond, data_shape=data_shape, ddim_sample=ddim_sample, *args, **kwargs)

    def get_action(self, cond, data_shape, ddim_sample, *args, **kwargs):
        return self.conditional_sample(cond, data_shape, ddim_sample, *args, **kwargs)

    def generate_base_model_parameters(self, argus, list_out=True):
        unet_parameters = self.model.generate_base_model_parameters(argus=argus, list_out=list_out)
        return unet_parameters