
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)   


class GaussianDiffusionSampler_DDIM(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, eta=0.0, steps=50):
        super().__init__()
        self.model = model
        self.T = T
        self.eta = eta
        self.steps = steps
        
        # comput β, α, α_bar
        betas = torch.linspace(beta_1, beta_T, T, dtype=torch.float32)
        self.register_buffer('betas', betas)
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        
        # register
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        self.register_buffer('alphas_bar', alphas_bar)
        
        # uniform step subsequence
        sampling_steps = torch.linspace(0, T-1, steps).round().long().unique()
        self.register_buffer('sampling_steps', sampling_steps)
        
        # comput α_bar (current and previous)
        prev_alpha_bar = F.pad(alphas_bar[sampling_steps[:-1]], (1, 0), value=1.0)
        self.register_buffer('prev_alpha_bar', prev_alpha_bar)
        self.register_buffer('current_alpha_bar', alphas_bar[sampling_steps])
        
        #  η σ_t
        sigma2_t = self.eta * (1 - prev_alpha_bar) / (1 - alphas_bar[sampling_steps]) * \
                   (1 - alphas_bar[sampling_steps] / prev_alpha_bar)
        sigma2_t[0] = 0.0
        sigma_t = torch.sqrt(sigma2_t)
        self.register_buffer('sigma_t', sigma_t)

    def predict_x0_from_eps(self, x_t, t, eps):
        """predict x_0 from noise"""
        return (x_t - extract(self.sqrt_one_minus_alphas_bar, t, x_t.shape) * eps
                ) / extract(self.sqrt_alphas_bar, t, x_t.shape)

    def forward(self, x_T):
        batch_size = x_T.shape[0]
        device = x_T.device
        
        # xt
        x_t = x_T.float()
        step_index = len(self.sampling_steps) - 1
        
        # sample from last step
        for step_index in range(len(self.sampling_steps) - 1, -1, -1):
            # current time step
            t = self.sampling_steps[step_index]
            t_tensor = t * torch.ones(batch_size, dtype=torch.long, device=device)
            print(t)

            # 1. noise ε_θ
            eps_theta = self.model(x_t, t_tensor)
            
            # 2. x_0
            x0_pred = self.predict_x0_from_eps(x_t, t_tensor, eps_theta)
            
            # 3. σ_t, α_bar_t, α_bar_t-1
            view_shape = (1,) * (len(x_t.shape) - 1)
            sigma_t = self.sigma_t[step_index].view(-1, *view_shape)
            prev_alpha_bar = self.prev_alpha_bar[step_index].view(-1, *view_shape)
            current_alpha_bar = self.current_alpha_bar[step_index].view(-1, *view_shape)
            
            # 4. direction pointing to x_t
            deterministic_part = torch.sqrt(prev_alpha_bar) * x0_pred
            
            # 5. var
            variance_weight = torch.sqrt(torch.clamp(1 - prev_alpha_bar - sigma_t**2, min=0.0))
            variance_part = variance_weight * eps_theta
            
            # 6. random noise
            if step_index > 0:
                random_noise = torch.randn_like(x_t)
                x_prev = deterministic_part + variance_part + sigma_t * random_noise
            else:
                x_prev = deterministic_part + variance_part
            
            x_t = x_prev
        
        return torch.clip(x_t, -1, 1)