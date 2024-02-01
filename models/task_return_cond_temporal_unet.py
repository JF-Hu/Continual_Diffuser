import torch
import torch.nn as nn
import einops
import math
from einops.layers.torch import Rearrange
from einops import rearrange
import pdb
from torch.distributions import Bernoulli
import numpy as np
from models.position_embedding import SinusoidalPosEmb
from config.hyperparameter import FineTuneParaType

class Downsample1d(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv(x)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.InstanceNorm2d(dim, affine = True)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 128):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

class GlobalMixing(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 128):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''
    def __init__(self, inp_channels, out_channels, kernel_size, mish=True, n_groups=8):
        super().__init__()

        if mish:
            act_fn = nn.Mish()
        else:
            act_fn = nn.SiLU()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            act_fn,
        )

    def forward(self, x):
        return self.block(x)

class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5, mish=True):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size, mish),
            Conv1dBlock(out_channels, out_channels, kernel_size, mish),
        ])

        if mish:
            act_fn = nn.Mish()
        else:
            act_fn = nn.SiLU()

        self.time_mlp = nn.Sequential(
            act_fn,
            nn.Linear(embed_dim, out_channels),
        )
        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        time_embed = self.time_mlp(t)
        if len(time_embed.shape) != len(x.shape):
            time_embed = einops.rearrange(time_embed, 'batch t -> batch t 1')
        else:
            time_embed = einops.rearrange(time_embed, 'batch h t -> batch t h')
        out = self.blocks[0](x) + time_embed
        out = self.blocks[1](out)
        return out + self.residual_conv(x)

class TaskReturnBasedTemporalUnet(nn.Module):

    def __init__(
            self, sequence_length, input_channels, out_channels, task_identity_dim=1, dim=64, dim_mults=(1, 2, 4, 8),
            kernel_size=5, condition_dropout=0.25,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.condition_dropout = condition_dropout
        dims = [input_channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'Temporal Unet Structure: Channel dimensions: {in_out}')
        mish = True
        act_fn = nn.Mish()

        self.time_dim = dim
        self.returns_dim = dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )

        self.task_identity_mlp = nn.Sequential(
            nn.Linear(task_identity_dim, dim),
            act_fn,
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )

        self.return_mlp = nn.Sequential(
            nn.Linear(task_identity_dim, dim),
            act_fn,
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )
        self.mask_dist = Bernoulli(probs=1 - self.condition_dropout)

        embed_dim = dim * 2

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        block_in_out_recoder = []

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=embed_dim, horizon=sequence_length, kernel_size=kernel_size, mish=mish),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=embed_dim, horizon=sequence_length, kernel_size=kernel_size, mish=mish),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))
            if not is_last:
                block_in_out_recoder.append((dim_in, dim_out, sequence_length, (sequence_length + 2 * 1 - 3)//2 + 1))
                sequence_length = block_in_out_recoder[-1][-1]
            else:
                block_in_out_recoder.append((dim_in, dim_out, sequence_length, sequence_length))

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=sequence_length, kernel_size=kernel_size, mish=mish)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=sequence_length, kernel_size=kernel_size, mish=mish)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            downs_input, downs_output, downs_input_length, downs_output_length = block_in_out_recoder.pop()
            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out + downs_output, downs_input, embed_dim=embed_dim, horizon=sequence_length, kernel_size=kernel_size, mish=mish),
                ResidualTemporalBlock(downs_input, downs_input, embed_dim=embed_dim, horizon=sequence_length, kernel_size=kernel_size, mish=mish),
                # Upsample1d(dim_in, *self.find_kernel_padding(down_input=downs_input, down_output=downs_output, stride=2)) if not is_last else nn.Identity()
                # Upsample1d(downs_input, kernel_size=block_in_out_recoder[-1][-2] - block_in_out_recoder[-1][-1]+2*1+1, stride=1, padding=1) if not is_last else nn.Identity()
                Upsample1d(downs_input) if not is_last else nn.Identity()
            ]))
            if not is_last:
                sequence_length = downs_input_length

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=kernel_size, mish=mish),
            nn.Conv1d(dim, out_channels, 1),
        )

    def find_kernel_padding(self, down_input, down_output, stride=2):
        for padding in range(0, down_input):
            kernel_size = down_output + 2 * padding - 2 * (down_input - 1)
            if kernel_size > 0 and kernel_size % 1 == 0:
                return (int(kernel_size), stride, padding)
        return None, None

    def forward(self, x, cond, timesteps, task_identity, returns, use_dropout=True, force_dropout=False, **kwargs):
        '''
            x : [ batch x sequence_length x transition ]
            returns : [batch x sequence_length]
        '''
        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(timesteps)
        # task_embed = self.task_identity_mlp(task_identity)
        return_embed = self.return_mlp(returns)

        if use_dropout:
            mask = self.mask_dist.sample(sample_shape=(return_embed.size(0), 1)).to(return_embed.device)
            return_embed = mask * return_embed
        if force_dropout:
            return_embed = 0 * return_embed
        t = torch.cat([t, return_embed], dim=-1)
        h = []

        idx = 0
        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)
            idx += 1
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)
        # import pdb; pdb.set_trace()
        for resnet, resnet2, upsample in self.ups:
            x_ = h.pop()
            x = torch.cat([x, x_], dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)
        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')

        return x

    def generate_base_model_parameters(self, argus, list_out=True):
        base_parameters = [] if list_out else {}
        for name, para in self.named_parameters():
            if "lora_A" not in name and "lora_B" not in name:
                if list_out:
                    base_parameters.append(para)
                else:
                    base_parameters.update({name: para})
        return base_parameters

    def generate_LoRA_parameters(self, argus, list_out=True):
        LoRA_parameters = [] if list_out else {}
        if argus.finetune_para_type in [FineTuneParaType.ResidualTemporalBlock_TimeMlp]:
            for name, module in self.named_modules():
                if isinstance(module, ResidualTemporalBlock):
                    if list_out:
                        LoRA_parameters.extend(module.time_mlp.parameters())
                    else:
                        for param_name, param in module.time_mlp.named_parameters():
                            LoRA_parameters[f"{name}.time_mlp.{param_name}"] = param
        elif argus.finetune_para_type == FineTuneParaType.TemporalDowns:
            for name, module in self.named_modules():
                if name == "downs":
                    if list_out:
                        LoRA_parameters.extend(module.parameters())
                    else:
                        for param_name, param in module.named_parameters():
                            LoRA_parameters[f"{name}.{param_name}"] = param
        else:
            raise NotImplementedError
        return LoRA_parameters

    def load_LoRA_parameters(self, argus, saved_parameters):
        if argus.finetune_para_type in [FineTuneParaType.ResidualTemporalBlock_TimeMlp]:
            for name, module in self.named_modules():
                if isinstance(module, ResidualTemporalBlock):
                    para_to_be_load = {}
                    for param_name, param in module.time_mlp.named_parameters():
                        para_to_be_load.update({f"{param_name}": saved_parameters[f"{name}.time_mlp.{param_name}"]})
                    module.time_mlp.load_state_dict(para_to_be_load)
        elif argus.finetune_para_type == FineTuneParaType.TemporalDowns:
            for name, module in self.named_modules():
                if name == "downs":
                    para_to_be_load = {}
                    for param_name, param in module.named_parameters():
                        para_to_be_load.update({f"{param_name}": saved_parameters[f"{name}.{param_name}"]})
                    module.load_state_dict(para_to_be_load)
        else:
            raise NotImplementedError

    def set_LoRA_network(self, argus):
        # todo There are something wrong about this func
        if argus.finetune_para_type in [FineTuneParaType.ResidualTemporalBlock_TimeMlp]:
            for name, module in self.named_modules():
                if isinstance(module, ResidualTemporalBlock):
                    for param_name, param in module.named_parameters():
                        if "time_mlp" in param_name:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                elif not isinstance(module, TaskBasedTemporalUnet):
                    for param_name, param in module.named_parameters():
                        param.requires_grad = False
        else:
            raise NotImplementedError