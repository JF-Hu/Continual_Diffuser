import torch
import torch.nn as nn
import einops
import math
from einops.layers.torch import Rearrange
from einops import rearrange
import pdb
from torch.distributions import Bernoulli
import numpy as np
from models.temporal_unet import SinusoidalPosEmb, Conv1dBlock, Downsample1d, Upsample1d, ResidualTemporalBlock
from models.lora_layer import LoRAConv1d, LoRAConvTranspose1d, LoRALinear
from config.hyperparameter import FineTuneParaType

class LoRAIdentity(nn.Module):
    def __init__(self):
        super(LoRAIdentity, self).__init__()
        self.identity = nn.Identity()
    def forward(self, x, task_name):
        return self.identity(x)

class LoRADownsample1d(nn.Module):
    def __init__(self, dim, task_names, lora_dim, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = LoRAConv1d(nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, stride=stride, padding=padding), task_name=task_names, r=lora_dim)

    def forward(self, x, task_name):
        return self.conv(x, task_name)

class LoRAUpsample1d(nn.Module):
    def __init__(self, dim, task_names, lora_dim, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.conv = LoRAConvTranspose1d(nn.ConvTranspose1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, stride=stride, padding=padding), task_name=task_names, r=lora_dim)

    def forward(self, x, task_name):
        return self.conv(x, task_name)

class Conv1dLoRABlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''
    def __init__(self, inp_channels, out_channels, kernel_size, task_names, lora_dim, mish=True, n_groups=8):
        super().__init__()

        if mish:
            act_fn = nn.Mish()
        else:
            act_fn = nn.SiLU()

        self.blocks = nn.ModuleList([
            LoRAConv1d(nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2), task_name=task_names, r=lora_dim),
            nn.Sequential(
                Rearrange('batch channels horizon -> batch channels 1 horizon'),
                nn.GroupNorm(n_groups, out_channels, affine=True),
                Rearrange('batch channels 1 horizon -> batch channels horizon'),
                act_fn,
            )
        ])

    def forward(self, x, task_name):
        x = self.blocks[0](x, task_name)
        x = self.blocks[1](x)
        return x

class LoRAResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, task_names, lora_dim, kernel_size=5, mish=True):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dLoRABlock(inp_channels=inp_channels, out_channels=out_channels, kernel_size=kernel_size, mish=mish, task_names=task_names, lora_dim=lora_dim),
            Conv1dBlock(inp_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, mish=mish),
        ])

        if mish:
            act_fn = nn.Mish()
        else:
            act_fn = nn.SiLU()

        self.time_mlp = nn.Sequential(
            act_fn,
            nn.Linear(embed_dim, out_channels),
        )
        # self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) if inp_channels != out_channels else nn.Identity()
        self.residual_conv = LoRAConv1d(nn.Conv1d(inp_channels, out_channels, 1), task_name=task_names, r=lora_dim) if inp_channels != out_channels else LoRAIdentity()

    def forward(self, x, t, task_name):
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
        out = self.blocks[0](x, task_name) + time_embed
        out = self.blocks[1](out)
        return out + self.residual_conv(x, task_name)

class LoRATemporalUnet(nn.Module):

    def __init__(
            self, sequence_length, input_channels, out_channels, task_names, lora_dim, dim=128, dim_mults=(1, 2, 4, 8),
            kernel_size=5,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.task_names = task_names
        dims = [input_channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'Temporal Unet Structure: Channel dimensions: {in_out}')
        mish = True
        act_fn = nn.Mish()

        self.time_dim = dim
        self.returns_dim = dim

        # self.time_mlp = nn.Sequential(
        #     SinusoidalPosEmb(dim),
        #     nn.Linear(dim, dim * 4),
        #     act_fn,
        #     nn.Linear(dim * 4, dim),
        # )
        self.time_emb = SinusoidalPosEmb(dim)
        self.time_mlp1 = LoRALinear(nn.Linear(dim, dim * 4), task_name=task_names, r=lora_dim)
        self.time_act1 = act_fn
        self.time_mlp2 = LoRALinear(nn.Linear(dim * 4, dim), task_name=task_names, r=lora_dim)

        embed_dim = dim

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        block_in_out_recoder = []

        # print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            down_lora_dim = lora_dim # int(dim_out/4)
            self.downs.append(nn.ModuleList([
                LoRAResidualTemporalBlock(dim_in, dim_out, embed_dim=embed_dim, horizon=sequence_length, kernel_size=kernel_size, mish=mish, task_names=task_names, lora_dim=down_lora_dim) if ind==0 else ResidualTemporalBlock(dim_in, dim_out, embed_dim=embed_dim, horizon=sequence_length, kernel_size=kernel_size, mish=mish, task_names=task_names, lora_dim=down_lora_dim),
                LoRAResidualTemporalBlock(dim_out, dim_out, embed_dim=embed_dim, horizon=sequence_length, kernel_size=kernel_size, mish=mish, task_names=task_names, lora_dim=down_lora_dim) if ind==0 else ResidualTemporalBlock(dim_out, dim_out, embed_dim=embed_dim, horizon=sequence_length, kernel_size=kernel_size, mish=mish, task_names=task_names, lora_dim=down_lora_dim),
                # LoRADownsample1d(dim_out, task_names=task_names, lora_dim=down_lora_dim) if not is_last else LoRAIdentity()
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))
            if not is_last:
                block_in_out_recoder.append((dim_in, dim_out, sequence_length, (sequence_length + 2 * 1 - 3)//2 + 1))
                sequence_length = block_in_out_recoder[-1][-1]
            else:
                block_in_out_recoder.append((dim_in, dim_out, sequence_length, sequence_length))

        mid_dim = dims[-1]
        mid_lora_dim = lora_dim # int(mid_dim/4)
        self.mid_block1 = LoRAResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=sequence_length, kernel_size=kernel_size, mish=mish, task_names=task_names, lora_dim=mid_lora_dim)
        self.mid_block2 = LoRAResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=sequence_length, kernel_size=kernel_size, mish=mish, task_names=task_names, lora_dim=mid_lora_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            downs_input, downs_output, downs_input_length, downs_output_length = block_in_out_recoder.pop()
            self.ups.append(nn.ModuleList([
                LoRAResidualTemporalBlock(dim_out + downs_output, downs_input, embed_dim=embed_dim, horizon=sequence_length, kernel_size=kernel_size, mish=mish, task_names=task_names, lora_dim=lora_dim) if ind == (len(in_out[1:])-1) else ResidualTemporalBlock(dim_out + downs_output, downs_input, embed_dim=embed_dim, horizon=sequence_length, kernel_size=kernel_size, mish=mish, task_names=task_names, lora_dim=lora_dim),
                LoRAResidualTemporalBlock(downs_input, downs_input, embed_dim=embed_dim, horizon=sequence_length, kernel_size=kernel_size, mish=mish, task_names=task_names, lora_dim=lora_dim) if ind == (len(in_out[1:])-1) else ResidualTemporalBlock(downs_input, downs_input, embed_dim=embed_dim, horizon=sequence_length, kernel_size=kernel_size, mish=mish, task_names=task_names, lora_dim=lora_dim),
                # LoRAUpsample1d(downs_input, task_names=task_names, lora_dim=lora_dim) if not is_last else LoRAIdentity()
                Upsample1d(downs_input) if not is_last else nn.Identity()
            ]))
            if not is_last:
                sequence_length = downs_input_length

        self.final_conv1 = Conv1dLoRABlock(inp_channels=dim, out_channels=dim, kernel_size=kernel_size, task_names=task_names, lora_dim=lora_dim, mish=mish)
        self.final_conv2 = LoRAConv1d(nn.Conv1d(dim, out_channels, 1), task_name=task_names, r=lora_dim)

    def find_kernel_padding(self, down_input, down_output, stride=2):
        for padding in range(0, down_input):
            kernel_size = down_output + 2 * padding - 2 * (down_input - 1)
            if kernel_size > 0 and kernel_size % 1 == 0:
                return (int(kernel_size), stride, padding)
        return None, None

    def forward(self, x, cond, timesteps, task_name, **kwargs):
        '''
            x : [ batch x sequence_length x transition ]
            returns : [batch x sequence_length]
        '''
        x = einops.rearrange(x, 'b h t -> b t h')

        # t = self.time_mlp(timesteps)
        t = self.time_emb(timesteps)
        t = self.time_act1(self.time_mlp1(t, task_name))
        t = self.time_mlp2(t, task_name)

        h = []
        idx = 0
        for block_idx, (resnet, resnet2, downsample) in enumerate(self.downs):
            x = resnet(x, t, task_name) if block_idx == 0 else resnet(x, t)
            x = resnet2(x, t, task_name) if block_idx == 0 else resnet2(x, t)
            h.append(x)
            x = downsample(x)
            idx += 1
        x = self.mid_block1(x, t, task_name)
        x = self.mid_block2(x, t, task_name)
        # import pdb; pdb.set_trace()
        for block_idx, (resnet, resnet2, upsample) in enumerate(self.ups):
            x_ = h.pop()
            x = torch.cat([x, x_], dim=1)
            x = resnet(x, t, task_name) if block_idx == (len(self.ups) -1) else resnet(x, t)
            x = resnet2(x, t, task_name) if block_idx == (len(self.ups) -1) else resnet2(x, t)
            x = upsample(x)
        x = self.final_conv1(x, task_name)
        x = self.final_conv2(x, task_name)

        x = einops.rearrange(x, 'b t h -> b h t')

        return x

    def generate_LoRA_parameters(self, argus, task_name, list_out=True):
        assert isinstance(task_name, str) and task_name in self.task_names
        LoRA_parameters = [] if list_out else {}
        if argus.finetune_para_type in [FineTuneParaType.InMidOutConv1d]:
            for name, para in self.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    if task_name in name:
                        if list_out:
                            LoRA_parameters.append(para)
                        else:
                            LoRA_parameters.update({name: para})
        else:
            raise NotImplementedError
        return LoRA_parameters

    def generate_base_model_parameters(self, argus, list_out=True):
        base_parameters = [] if list_out else {}
        for name, para in self.named_parameters():
            if "lora_A" not in name and "lora_B" not in name:
                if list_out:
                    base_parameters.append(para)
                else:
                    base_parameters.update({name: para})
        return base_parameters

    def control_base_model_training(self, train_mode):
        for name, para in self.named_parameters():
            if "lora_A" not in name and "lora_B" not in name:
                para.requires_grad_(train_mode)

    def control_lora_model_training(self, task_name, train_mode):
        assert isinstance(task_name, str) and task_name in self.task_names
        for name, para in self.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                if task_name in name:
                    para.requires_grad_(train_mode)
