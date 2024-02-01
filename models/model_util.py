import copy
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod

def to_np(x):
	if torch.is_tensor(x):
		x = x.detach().cpu().numpy()
	return x

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

def apply_conditioning(x, conditions, action_dim, slide_mode=False, sr_sequence=False):
    # todo if slide_mode, which means that we condition on the history t sub sequences
    if slide_mode:
        for t, val in conditions.items():
            if sr_sequence:
                temp_reward = copy.deepcopy(x[:, t, -1])
                if len(val.shape) < len(x[:, 0:t + 1, action_dim:].shape):
                    x[:, 0:t + 1, action_dim:] = torch.unsqueeze(val.clone(), dim=-2)
                else:
                    x[:, 0:t + 1, action_dim:] = val.clone()
                x[:, t, -1] = temp_reward
            else:
                if len(val.shape) < len(x[:, 0:t+1, action_dim:].shape):
                    x[:, 0:t+1, action_dim:] = torch.unsqueeze(val.clone(), dim=-2)
                else:
                    x[:, 0:t+1, action_dim:] = val.clone()
    else:
        assert not sr_sequence
        for t, val in conditions.items():
            x[:, t, action_dim:] = val.clone()
    return x

def replaceable_condition(x, conditions, start_dim, end_dim):
    for t, val in conditions.items():
        x[:, t, start_dim:end_dim] = val.clone()
    return x

def apply_conditioning_diffusion_q_function(x, conditions, action_dim):
    # todo if slide_mode, which means that we condition on the history t sub sequences
    for t, val in conditions.items():
        x[:, :, action_dim:-1] = val.clone()
    return x

def apply_first_item_conditioning(x, conditions, fixed_dim):
    # todo if slide_mode, which means that we condition on the history t sub sequences
    for t, val in conditions.items():
        x[:, t, :fixed_dim] = val.clone()
    return x

def cancatnate_condition(x, conditions):
    x = torch.cat([x, conditions], dim=-1)
    return x


class WeightedLoss(nn.Module):

    def __init__(self, weights, action_dim):
        super().__init__()
        self.register_buffer('weights', weights)
        self.action_dim = action_dim

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        a0_loss = (loss[:, 0, :self.action_dim] / self.weights[0, :self.action_dim]).mean()
        return weighted_loss, {'a0_loss': a0_loss}

class NoneWeightedLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss = self._loss(pred, targ)
        loss = loss.mean()
        return loss, {'diffusion_loss': loss.detach().cpu().numpy()}

class WeightedStateLoss(nn.Module):

    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', weights)

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        return weighted_loss, {'state_loss': weighted_loss}

class ValueLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        pass

    def forward(self, pred, targ):
        loss = self._loss(pred, targ).mean()

        if len(pred) > 1:
            corr = np.corrcoef(
                to_np(pred).squeeze(),
                to_np(targ).squeeze()
            )[0,1]
        else:
            corr = np.NaN

        info = {
            'mean_pred': pred.mean(), 'mean_targ': targ.mean(),
            'min_pred': pred.min(), 'min_targ': targ.min(),
            'max_pred': pred.max(), 'max_targ': targ.max(),
            'corr': corr,
        }

        return loss, info

class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class BCL2(NoneWeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class NormalL2(NoneWeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class WeightedStateL2(WeightedStateLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class ValueL1(ValueLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class ValueL2(ValueLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
    'bc_l2': BCL2,
    'state_l2': WeightedStateL2,
    'value_l1': ValueL1,
    'value_l2': ValueL2,
    'NormalL2': NormalL2,
}

def _to_str(num):
	if num >= 1e6:
		return f'{(num/1e6):.2f} M'
	else:
		return f'{(num/1e3):.2f} k'

def param_to_module(param):
	module_name = param[::-1].split('.', maxsplit=1)[-1][::-1]
	return module_name

def report_parameters(model, topk=10):
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
		print(' '*8, f'{key:10}: {_to_str(count)} | {modules[module]}')

	remaining_parameters = sum([counts[k] for k in sorted_keys[topk:]])
	print(' '*8, f'... and {len(counts)-topk} others accounting for {_to_str(remaining_parameters)} parameters')
	return n_parameters

# ############################################ SpatialTransformerUnet functions ########################################

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def normalization(channels, n_groups=8):
    assert n_groups in [8, 16, 32]
    return GroupNorm32(n_groups, channels)

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def conv_transpose_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.ConvTranspose1d(*args, **kwargs)
    elif dims == 2:
        return nn.ConvTranspose2d(*args, **kwargs)
    elif dims == 3:
        return nn.ConvTranspose3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param convolution_type: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_transpose_conv, convolution_type=1, out_channels=None, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_transpose_conv = use_transpose_conv
        self.convolution_type = convolution_type
        if use_transpose_conv:
            self.conv = conv_transpose_nd(convolution_type, self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = conv_nd(self.convolution_type, self.channels, self.out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        # if self.dims == 3:
        #     x = F.interpolate(
        #         x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
        #     )
        # else:
        #     x = F.interpolate(x, scale_factor=2, mode="nearest")
        if not self.use_transpose_conv:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x

def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, convolution_type=1, out_channels=None, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.convolution_type = convolution_type
        # stride = 2 if convolution_type != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                convolution_type, self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(convolution_type, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])

class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

def calculate_convolution_parameter_setting(H_in=None, H_out=None, kernel_size=None, stride=None, padding=None):
    convolution_parameter_setting = [H_in, H_out, kernel_size, stride, padding]
    assert convolution_parameter_setting.count(None) == 1
    if H_in is None:
        return (H_out - 1) * stride + kernel_size - 2 * padding
    if H_out is None:
        return (H_in + 2 * padding - kernel_size) // stride + 1
    if kernel_size is None:
        return H_in + 2 * padding - (H_out - 1) * stride
    if stride is None:
        out = (H_in + 2 * padding - kernel_size) / (H_out - 1)
        if out == int(out):
            return out
        else:
            raise Exception("convolution parameter setting [stride] doesn't match the given parameters.")
    if padding is None:
        out = ((H_out - 1) * stride + kernel_size - H_in)/2
        if out == int(out):
            return out
        else:
            raise Exception("convolution parameter setting [padding] doesn't match the given parameters.")
