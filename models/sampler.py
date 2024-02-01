import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from models.model_util import extract
from trainer.trainer_util import batch_to_device

class DDIMSampler_original():

    def __init__(self, alphas, alphas_cumprod, stride=10, eta=1):
        self.eta = eta
        self.alphas = alphas
        self.bar_alpha = alphas_cumprod
        self.bar_alpha_ = self.bar_alpha[::stride]
        self.bar_alpha_pre_ = F.pad(self.bar_alpha_[:-1], [1, 0], mode='constant', value=1)

        self.bar_beta_ = 1 - self.bar_alpha_
        self.bar_beta_pre_ = 1 - self.bar_alpha_pre_
        self.alpha_ = torch.sqrt(self.bar_alpha_ / self.bar_alpha_pre_)
        self.sigma_ = torch.sqrt(self.bar_beta_pre_ / self.bar_beta_) * torch.sqrt(1 - self.bar_alpha_ / self.bar_alpha_pre_) * self.eta
        self.log_sigma_ = torch.log(torch.clamp(self.sigma_, min=1e-20))
        # self.epsilon_ = torch.sqrt(self.bar_beta_) - self.alpha_ * torch.sqrt(self.bar_beta_pre_ - self.sigma_ ** 2)
        self.epsilon_ = (1 - self.bar_alpha_ / self.bar_alpha_pre_) / torch.sqrt(1 - self.bar_alpha_)
        self.generation_step = len(self.bar_alpha_)

    def sample_one_step(self, epsilon, x_t, t):

        b, *_, device = *x_t.shape, x_t.device
        noise = 0.5*torch.randn_like(x_t)
        t = torch.full((b,), t, device=device, dtype=torch.long)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_t.shape) - 1)))

        self.epsilon_ = self.epsilon_.to(device)
        self.alpha_ = self.alpha_.to(device)
        self.sigma_ = self.sigma_.to(device)
        self.log_sigma_ = self.log_sigma_.to(device)
        x_t = x_t - extract(self.epsilon_, t, x_t.shape) * epsilon
        x_t = x_t / extract(self.alpha_, t, x_t.shape)
        model_mean = x_t
        posterior_variance = extract(self.sigma_, t, x_t.shape)
        model_log_variance = extract(self.log_sigma_, t, x_t.shape)
        x_t_1 = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return x_t_1

class DDIMSampler():

    def __init__(self, alphas, alphas_cumprod, stride=10, eta=1, betas=None, alphas_cumprod_prev=None, posterior_variance=None):
        self.eta = eta
        self.alphas = alphas
        self.bar_alpha = alphas_cumprod
        self.bar_alpha_ = self.bar_alpha[::stride]
        self.bar_alpha_pre_ = F.pad(self.bar_alpha_[:-1], [1, 0], mode='constant', value=1)

        self.bar_beta_ = 1 - self.bar_alpha_
        self.bar_beta_pre_ = 1 - self.bar_alpha_pre_
        self.alpha_ = torch.sqrt(self.bar_alpha_ / self.bar_alpha_pre_)
        # self.sigma_ = (1 - self.bar_alpha_ / self.bar_alpha_pre_) * self.bar_beta_pre_ / self.bar_beta_ * self.eta
        self.sigma_ = (self.bar_alpha_pre_ - self.bar_alpha_) * self.bar_beta_pre_ / (self.bar_beta_ * self.bar_alpha_pre_) * self.eta
        self.log_sigma_ = torch.log(torch.clamp(self.sigma_, min=1e-20))
        # self.epsilon_ = torch.sqrt(self.bar_beta_) - self.alpha_ * torch.sqrt(self.bar_beta_pre_ - self.sigma_)
        self.epsilon_ = torch.sqrt(self.bar_beta_) - torch.sqrt((self.bar_beta_pre_ - self.sigma_)*self.bar_alpha_ / self.bar_alpha_pre_)
        self.generation_step = len(self.bar_alpha_)

    def sample_one_step(self, epsilon, x_t, t, rl_time=None, **kwargs):
        b, *_, device = *x_t.shape, x_t.device
        noise = torch.randn_like(x_t)
        t = torch.full((b,), t, device=device, dtype=torch.long)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_t.shape) - 1)))

        self.epsilon_ = self.epsilon_.to(device)
        self.alpha_ = self.alpha_.to(device)
        self.sigma_ = self.sigma_.to(device)
        self.log_sigma_ = self.log_sigma_.to(device)
        x_t = x_t - extract(self.epsilon_, t, x_t.shape) * epsilon
        x_t = x_t / extract(self.alpha_, t, x_t.shape)
        model_mean = x_t
        posterior_variance = extract(self.sigma_, t, x_t.shape)
        model_log_variance = extract(self.log_sigma_, t, x_t.shape)
        x_t_1 = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return x_t_1

class AnalyticDPMSampler():

    def __init__(self, alphas, betas, alphas_cumprod_prev, posterior_variance, alphas_cumprod, stride=10, eta=1):
        self.factors = []
        self.factors_flag = False
        self.eta = eta
        self.alphas = alphas
        self.bar_alpha = alphas_cumprod
        self.stride = stride
        self.bar_alpha_ = self.bar_alpha[::stride]
        self.bar_alpha_pre_ = F.pad(self.bar_alpha_[:-1], [1, 0], mode='constant', value=1)

        self.bar_beta_ = 1 - self.bar_alpha_
        self.bar_beta_pre_ = 1 - self.bar_alpha_pre_
        self.alpha_ = torch.sqrt(self.bar_alpha_ / self.bar_alpha_pre_)
        # self.sigma_ = (1 - self.bar_alpha_ / self.bar_alpha_pre_) * self.bar_beta_pre_ / self.bar_beta_ * self.eta
        self.sigma_square_ = (self.bar_alpha_pre_ - self.bar_alpha_) * self.bar_beta_pre_ / (self.bar_beta_ * self.bar_alpha_pre_) * self.eta
        self.log_sigma_square_ = torch.log(torch.clamp(self.sigma_square_, min=1e-20))
        # self.epsilon_ = torch.sqrt(self.bar_beta_) - self.alpha_ * torch.sqrt(self.bar_beta_pre_ - self.sigma_)
        self.epsilon_ = torch.sqrt(self.bar_beta_) - torch.sqrt((self.bar_beta_pre_ - self.sigma_square_)*self.bar_alpha_ / self.bar_alpha_pre_)
        self.gamma_times_beta_alpha = self.epsilon_ * self.bar_alpha_pre_ / self.bar_alpha_
        self.generation_step = len(self.bar_alpha_)

    def calculate_analytic_dpm_sigma(self, diffusion_model, dataloader, original_diffusion_steps, return_type, action_dim, q_sample, horizon, returns, goals, predict_epsilon, device):
        assert predict_epsilon and not self.factors_flag
        for t in range(original_diffusion_steps):
            with torch.no_grad():
                batch = next(dataloader)
                batch = batch_to_device(batch, device=device, convert_to_torch_float=True)
                if return_type == 21:
                    x_start = batch.trajectories[:, :, :]
                else:
                    x_start = batch.trajectories[:, :, action_dim:]
                cond = batch.conditions
                noise = torch.randn_like(x_start)
                batch_size = len(x_start)
                diffusion_step = torch.full((batch_size,), t, dtype=torch.long, device=x_start.device)
                # t = torch.randint(0, original_diffusion_steps, (batch_size,), device=x_start.device).long()
                x_noisy = q_sample(x_start=x_start, t=diffusion_step, noise=noise)
                if return_type == 21:
                    apply_condition_offset = action_dim
                    x_noisy = apply_sa_conditioning(x_noisy, cond, apply_condition_offset)
                else:
                    apply_condition_offset = 0
                    x_noisy = apply_conditioning(
                        x_noisy, cond, apply_condition_offset, horizon=horizon,
                        slide_mode=True if return_type == 11 or return_type == 13 or return_type == 15 or return_type == 18 or return_type == 19 or return_type == 20 else False,
                        sr_sequence=True if return_type == 19 else False,
                        goal_state_cond=True if return_type == 22 else False,
                    )
                x_recon = diffusion_model.model(x_noisy, cond, diffusion_step, returns, goals=goals)
                ## todo Here, we should remove the cond to cut off the influence of cond, but we here to neglect this step!!!
            self.factors.append((x_recon**2).mean())
        self.factors = torch.reshape(torch.tensor(self.factors), self.alphas.shape)
        self.factors.to('cpu')
        self.factors = torch.clamp(1 - self.factors, 0, 1)
        self.sigma_square_ = self.sigma_square_ ** 2 + self.gamma_times_beta_alpha ** 2 * self.factors[::self.stride]
        self.log_sigma_square_ = torch.log(torch.clamp(self.sigma_square_, min=1e-20))
        self.factors_flag = True

    def sample_one_step(self, epsilon, x_t, t):
        # if not self.factors_flag:
        #     raise Exception("You should reload diffusion model parameters and call the function 'calculate_analytic_dpm_sigma()' to generate factors")
        b, *_, device = *x_t.shape, x_t.device
        noise = torch.randn_like(x_t)
        t = torch.full((b,), t, device=device, dtype=torch.long)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_t.shape) - 1)))

        self.epsilon_ = self.epsilon_.to(device)
        self.alpha_ = self.alpha_.to(device)
        self.sigma_square_ = self.sigma_square_.to(device)
        self.log_sigma_square_ = self.log_sigma_square_.to(device)
        x_t = x_t - extract(self.epsilon_, t, x_t.shape) * epsilon
        x_t = x_t / extract(self.alpha_, t, x_t.shape)
        model_mean = x_t
        posterior_variance = extract(self.sigma_square_, t, x_t.shape)
        model_log_variance = extract(self.log_sigma_square_, t, x_t.shape)
        x_t_1 = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return x_t_1








