import torch
import math
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.odd_dim_flag = self.dim % 2
        if self.odd_dim_flag:
            self.half_dim = self.dim // 2 + 1
        else:
            self.half_dim = self.dim // 2
        # self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = math.log(10000) / self.half_dim

    def forward(self, x):
        device = x.device
        sin_emb = torch.exp(torch.arange(self.half_dim, device=device) * -self.emb)
        cos_emb = torch.exp(torch.arange(self.half_dim - 1, device=device) * -self.emb) if self.odd_dim_flag else torch.exp(torch.arange(self.half_dim, device=device) * -self.emb)
        if len(x.shape) == 1:
            sin_emb = x[:, None] * sin_emb[None, :]
            cos_emb = x[:, None] * cos_emb[None, :]
        else:
            sin_emb = x.unsqueeze(-1) * sin_emb[None, None, :]
            cos_emb = x.unsqueeze(-1) * cos_emb[None, None, :]
        # emb = x[:, None] * emb[None, :]
        emb = torch.cat((sin_emb.sin(), cos_emb.cos()), dim=-1)
        return emb