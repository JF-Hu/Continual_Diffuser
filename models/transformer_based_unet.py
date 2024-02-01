import math
import torch
import numpy as np
from torch import nn, einsum
from inspect import isfunction
import torch.nn.functional as F
from einops import rearrange, repeat
from abc import abstractmethod
from torch.distributions import Bernoulli
from models.position_embedding import SinusoidalPosEmb
from models.model_util import TimestepBlock, normalization, conv_nd, Upsample, Downsample, calculate_convolution_parameter_setting
from config.hyperparameter import FineTuneParaType


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
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


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None, calc_energy=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        self.norm = Normalize(in_channels)

        self.proj_in = conv_nd(1, in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim) for d in range(depth)])

        self.proj_out = conv_nd(1, inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h -> b h c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b h c -> b c h', h=h)
        x = self.proj_out(x)
        return x + x_in

class AlterableHeadDimTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None, calc_energy=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        inner_dim = n_heads * d_head

        self.norm_in = Normalize(in_channels)

        self.norm_out = Normalize(inner_dim)

        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim) for d in range(depth)])

        self.proj_out = nn.Conv1d(inner_dim*2, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h = x.shape
        x_in = x
        x = self.norm_in(x)
        x = rearrange(x, 'b c h -> b h c')
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b h c -> b c h', h=h)
        x_tf_out = x = x + x_in
        x = self.norm_out(x)
        x = self.proj_out(torch.cat([x, x_tf_out], dim=1))
        return x

class ResBlock(TimestepBlock):
    def __init__(
        self,
        input_channels,
        time_emb_channels,
        dropout,
        out_channels,
        use_conv=False,
        use_scale_shift_norm=False,
        convolution_type=1,
        use_checkpoint=False,
        up=False,
        down=False,
        kernel_size=None,
        stride=None,
        padding=None,
        same_conv_kernel_size=3,
        use_transpose_conv=True,
    ):
        super().__init__()
        self.channels = input_channels
        self.time_emb_channels = time_emb_channels
        self.dropout = dropout
        self.out_channels = out_channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.same_conv_kernel_size = same_conv_kernel_size
        self.use_transpose_conv = use_transpose_conv

        self.in_layers = nn.Sequential(
            normalization(input_channels),
            nn.SiLU(),
            conv_nd(convolution_type, input_channels, self.out_channels, self.same_conv_kernel_size, padding=self.same_conv_kernel_size//2),
        )

        self.updown = up or down

        if up:
            assert kernel_size is not None and stride is not None and padding is not None
            if use_transpose_conv:
                kernel_size, stride, padding = 4, 2, 1
            else:
                kernel_size, stride, padding = 3, 1, 1
            self.h_upd = Upsample(input_channels, use_transpose_conv, convolution_type=1, kernel_size=kernel_size, stride=stride, padding=padding)
            self.x_upd = Upsample(input_channels, use_transpose_conv, convolution_type=1, kernel_size=kernel_size, stride=stride, padding=padding)
        elif down:
            assert kernel_size is not None and stride is not None and padding is not None
            self.h_upd = Downsample(input_channels, True, convolution_type=1, kernel_size=kernel_size, stride=stride, padding=padding)
            self.x_upd = Downsample(input_channels, True, convolution_type=1, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.time_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels,),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            conv_nd(convolution_type, self.out_channels, self.out_channels, self.same_conv_kernel_size, padding=self.same_conv_kernel_size//2),
        )

        if self.out_channels == input_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(convolution_type, input_channels, self.out_channels, self.same_conv_kernel_size, padding=self.same_conv_kernel_size//2)
        else:
            self.skip_connection = conv_nd(convolution_type, input_channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x time_emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.time_emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

    def get_convolution_setting_recursively(self, module):
        convolution_inout_recoder = []
        for each_module in module._modules:
            if len(module._modules[each_module]._modules) > 0:
                out = self.get_convolution_setting_recursively(module._modules[each_module])
                # convolution_inout_recoder.extend(out)
            else:
                if isinstance(module._modules[each_module], nn.Conv1d):
                    out = [
                        {"in_channel": module._modules[each_module].in_channels, "out_channel": module._modules[each_module].out_channels,
                         "kernel_size": module._modules[each_module].kernel_size[0],
                         "stride": module._modules[each_module].stride[0],
                         "padding": module._modules[each_module].padding[0]
                         }
                    ]
                else:
                    out = []
            convolution_inout_recoder.extend(out)
        return convolution_inout_recoder

    def calculate_convolution_setting(self, in_length):
        result = self.get_convolution_setting_recursively(module=self)
        for i, each_result in enumerate(result):
            each_result.update(
                {"in_length": in_length,
                 "out_length": calculate_convolution_parameter_setting(H_in=in_length, H_out=None, kernel_size=each_result["kernel_size"], stride=each_result["stride"], padding=each_result["padding"]),
                 })
            in_length = each_result["out_length"]
            result[i] = each_result
        return result

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer) or isinstance(layer, AlterableHeadDimTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x

class SpatialTransformerUnet(nn.Module):
    def __init__(
            self, input_channels, sequence_length, out_channels, num_res_blocks, attention_resolutions, num_heads, model_channels,
            use_fp16=False, channel_mult=(1, 2, 4, 8), context_dim=None, dropout=0., per_head_channels=-1,
            legacy=True, use_spatial_transformer=True, transformer_depth=1, same_conv_kernel_size=5):
        super().__init__()
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.context_dim = context_dim
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.dropout = dropout
        self.predict_codebook_ids = False
        self.convolution_inout_recoder = []
        self.in_length = sequence_length
        self.transformer_depth = transformer_depth
        self.same_conv_kernel_size = same_conv_kernel_size

        if num_heads == -1:
            assert per_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if per_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        mish = True
        act_fn = nn.Mish()

        time_embed_dim = model_channels
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            act_fn,
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        context_embed_dim = model_channels
        self.context_mlp = nn.Sequential(
            nn.Linear(self.context_dim, model_channels),
            act_fn,
            nn.Linear(model_channels, context_embed_dim),
        )
        self.context_mask_dist = Bernoulli(probs=1 - 0.25)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(conv_nd(1, input_channels, model_channels, self.same_conv_kernel_size, padding=self.same_conv_kernel_size//2))
            ]
        )
        self.recoder_convolution_info(block=self.input_blocks[-1])

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        input_channels=ch, time_emb_channels=time_embed_dim, dropout=dropout, out_channels=mult * model_channels,
                        convolution_type=1, use_checkpoint=False, use_scale_shift_norm=False, same_conv_kernel_size=3)
                ]
                # aa = layers[0].calculate_convolution_setting(in_length=self.convolution_inout_recoder[-1]["out_length"])
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if per_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // per_head_channels
                        dim_head = per_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else per_head_channels
                    layers.append(SpatialTransformer(in_channels=ch, n_heads=num_heads, d_head=dim_head, depth=transformer_depth, context_dim=context_embed_dim))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.recoder_convolution_info(block=self.input_blocks[-1])
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            input_channels=ch, time_emb_channels=time_embed_dim, dropout=dropout, out_channels=out_ch,
                            convolution_type=1, use_checkpoint=False, use_scale_shift_norm=False, down=True,
                            kernel_size=3, stride=2, padding=1, same_conv_kernel_size=3),
                        # Downsample(ch, True, convolution_type=1, out_channels=out_ch, kernel_size=3, stride=2, padding=1,)
                    )
                )
                self.recoder_convolution_info(block=self.input_blocks[-1])
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if per_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // per_head_channels
            dim_head = per_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else per_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                input_channels=ch, time_emb_channels=time_embed_dim, dropout=0, out_channels=ch,
                convolution_type=1, use_checkpoint=False, use_scale_shift_norm=False, same_conv_kernel_size=self.same_conv_kernel_size),
            SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_embed_dim),
            ResBlock(
                input_channels=ch, time_emb_channels=time_embed_dim, dropout=0, out_channels=ch,
                convolution_type=1, use_checkpoint=False, use_scale_shift_norm=False, same_conv_kernel_size=self.same_conv_kernel_size),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        out_block_in_length = self.convolution_inout_recoder[-1][-1]["out_length"]
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                convolution_recoder = self.convolution_inout_recoder.pop()
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        input_channels=ch + ich, time_emb_channels=time_embed_dim, dropout=0, out_channels=model_channels * mult,
                        convolution_type=1, use_checkpoint=False, use_scale_shift_norm=False, same_conv_kernel_size=self.same_conv_kernel_size)
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if per_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // per_head_channels
                        dim_head = per_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else per_head_channels
                    layers.append(SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_embed_dim))
                if len(self.convolution_inout_recoder)>0 and convolution_recoder[-1]["out_length"] != self.convolution_inout_recoder[-1][-1]["out_length"]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            input_channels=ch, time_emb_channels=time_embed_dim, dropout=dropout, out_channels=out_ch,
                            convolution_type=1, use_checkpoint=False, use_scale_shift_norm=False, up=True,
                            kernel_size=3, stride=2, padding=1, same_conv_kernel_size=3, use_transpose_conv=True),
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(1, model_channels, out_channels, 3, padding=1),
        )

        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
                normalization(ch),
                conv_nd(1, model_channels, None, 1),
                # nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
            )

    def forward(self, x, cond, timesteps, context=None, use_dropout=True, force_dropout=False, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # assert (y is not None) == (self.num_classes is not None), "must specify y if and only if the model is class-conditional"
        hs = []
        x = rearrange(x, 'b h c -> b c h')
        time_emb = self.time_mlp(timesteps.to(x.device))
        context_emb = self.context_mlp(context.to(x.device))
        if use_dropout:
            mask = self.context_mask_dist.sample(sample_shape=(context_emb.size(0), 1)).to(context_emb.device)
            context_emb = mask * context_emb
        if force_dropout:
            context_emb = 0 * context_emb
        context_emb = torch.unsqueeze(context_emb, dim=1)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, time_emb, context_emb)
            hs.append(h)
        h = self.middle_block(h, time_emb, context_emb)
        for module in self.output_blocks:
            res_h = hs.pop()
            h = torch.cat([h, res_h], dim=1)
            h = module(h, time_emb, context_emb)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            h =  self.id_predictor(h)
        else:
            h = self.out(h)
        h = rearrange(h, 'b t h -> b h t')

        return h

    def get_convolution_setting_recursively(self, module):
        convolution_inout_recoder = []
        for module_name in module._modules:
            if module_name != "x_upd":
                if len(module._modules[module_name]._modules) > 0:
                    out = self.get_convolution_setting_recursively(module._modules[module_name])
                    # convolution_inout_recoder.extend(out)
                else:
                    if isinstance(module._modules[module_name], nn.Conv1d):
                        out = [
                            {"in_channel": module._modules[module_name].in_channels,
                             "out_channel": module._modules[module_name].out_channels,
                             "kernel_size": module._modules[module_name].kernel_size[0],
                             "stride": module._modules[module_name].stride[0],
                             "padding": module._modules[module_name].padding[0]
                             }
                        ]
                    else:
                        out = []
            else:
                out = []
            convolution_inout_recoder.extend(out)
        return convolution_inout_recoder

    def calculate_convolution_setting(self, block, in_length):
        result = self.get_convolution_setting_recursively(module=block)
        for i, each_result in enumerate(result):
            each_result.update(
                {"in_length": in_length,
                 "out_length": calculate_convolution_parameter_setting(H_in=in_length, H_out=None, kernel_size=each_result["kernel_size"], stride=each_result["stride"], padding=each_result["padding"]),
                 })
            in_length = each_result["out_length"]
            result[i] = each_result
        return result

    def recoder_convolution_info(self, block):
        out = self.calculate_convolution_setting(block=block, in_length=self.in_length)
        self.convolution_inout_recoder.append(out)
        self.in_length = out[-1]["out_length"]

class PurelyTransformerUnet(nn.Module):
    def __init__(
            self, input_channels, sequence_length, out_channels, num_res_blocks, attention_resolutions, num_heads, model_channels,
            use_fp16=False, channel_mult=(1, 2, 4, 8), context_dim=None, condition_dropout=0.1, num_head_channels=-1,
            legacy=True, use_spatial_transformer=True, transformer_depth=1, num_classes=None, same_conv_kernel_size=5):
        super().__init__()
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.context_dim = context_dim
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.condition_dropout = condition_dropout
        self.predict_codebook_ids = False
        self.num_classes = num_classes
        self.convolution_inout_recoder = []
        self.in_length = sequence_length
        self.transformer_depth = transformer_depth
        self.same_conv_kernel_size = same_conv_kernel_size

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        mish = True
        act_fn = nn.Mish()

        time_embed_dim = model_channels
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            act_fn,
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        returns_mlp_input_dim = 1
        return_embed_dim = self.context_dim * self.sequence_length
        self.returns_mlp = nn.Sequential(
            nn.Linear(returns_mlp_input_dim, model_channels),
            act_fn,
            nn.Linear(model_channels, return_embed_dim),
        )
        # self.mask_dist = Bernoulli(probs=1 - self.condition_dropout)

        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(1, input_channels, model_channels, self.same_conv_kernel_size, padding=self.same_conv_kernel_size // 2))])

        self._feature_size = model_channels
        ch = model_channels
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                ch = model_channels
                if num_head_channels == -1:
                    dim_head = ch // num_heads
                else:
                    num_heads = ch // num_head_channels
                    dim_head = num_head_channels
                if legacy:
                    dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

                layers = [AlterableHeadDimTransformer(in_channels=ch, out_channels=ch, n_heads=num_heads, d_head=dim_head, depth=transformer_depth, context_dim=context_dim)]
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            AlterableHeadDimTransformer(in_channels=ch, out_channels=ch, n_heads=num_heads, d_head=dim_head, depth=transformer_depth, context_dim=context_dim),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks):
                ch = model_channels * 2
                if num_head_channels == -1:
                    dim_head = ch // num_heads
                else:
                    num_heads = ch // num_head_channels
                    dim_head = num_head_channels
                if legacy:
                    dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

                layers = [AlterableHeadDimTransformer(in_channels=ch, out_channels=model_channels, n_heads=num_heads, d_head=dim_head, depth=transformer_depth, context_dim=context_dim)]
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            conv_nd(1, model_channels, out_channels, 3, padding=1),
        )

        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
                normalization(ch),
                conv_nd(1, model_channels, None, 1),
                # nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
            )

    def forward(self, x, cond=None, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # assert (y is not None) == (self.num_classes is not None), "must specify y if and only if the model is class-conditional"
        hs = []
        x = rearrange(x, 'b h c -> b c h')
        time_emb = self.time_mlp(timesteps.to(x.device))
        context_emb = self.returns_mlp(context.to(x.device))
        context_emb = torch.reshape(context_emb, [len(x), self.sequence_length, -1])

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            time_emb = time_emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, time_emb, context_emb)
            hs.append(h)
        h = self.middle_block(h, time_emb, context_emb)
        for module in self.output_blocks:
            res_h = hs.pop()
            h = torch.cat([h, res_h], dim=1)
            h = module(h, time_emb, context_emb)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            h =  self.id_predictor(h)
        else:
            h = self.out(h)
        h = rearrange(h, 'b t h -> b h t')

        return h

    def get_convolution_setting_recursively(self, module):
        convolution_inout_recoder = []
        for module_name in module._modules:
            if module_name != "x_upd":
                if len(module._modules[module_name]._modules) > 0:
                    out = self.get_convolution_setting_recursively(module._modules[module_name])
                    # convolution_inout_recoder.extend(out)
                else:
                    if isinstance(module._modules[module_name], nn.Conv1d):
                        out = [
                            {"in_channel": module._modules[module_name].in_channels,
                             "out_channel": module._modules[module_name].out_channels,
                             "kernel_size": module._modules[module_name].kernel_size[0],
                             "stride": module._modules[module_name].stride[0],
                             "padding": module._modules[module_name].padding[0]
                             }
                        ]
                    else:
                        out = []
            else:
                out = []
            convolution_inout_recoder.extend(out)
        return convolution_inout_recoder

    def calculate_convolution_setting(self, block, in_length):
        result = self.get_convolution_setting_recursively(module=block)
        for i, each_result in enumerate(result):
            each_result.update(
                {"in_length": in_length,
                 "out_length": calculate_convolution_parameter_setting(H_in=in_length, H_out=None, kernel_size=each_result["kernel_size"], stride=each_result["stride"], padding=each_result["padding"]),
                 })
            in_length = each_result["out_length"]
            result[i] = each_result
        return result

    def recoder_convolution_info(self, block):
        out = self.calculate_convolution_setting(block=block, in_length=self.in_length)
        self.convolution_inout_recoder.append(out)
        self.in_length = out[-1]["out_length"]

if __name__ == "__main__":
    SUNET = SpatialTransformerUnet(
        input_channels=11, sequence_length=48, out_channels=11, num_res_blocks=1, attention_resolutions=(4,2,1), num_heads=8, model_channels=128, context_dim=1)
    SUNET.to("cuda")
    x = torch.tensor(np.ones((32, 48, 11))).to("cuda").to(torch.float32)
    bb = SUNET(x=x, cond=None, timesteps=(torch.reshape(torch.tensor(np.ones((32,))), [-1,])).to(torch.float32),
               context=(torch.reshape(torch.tensor(np.ones((32, 1))), [-1, 1])).to(torch.float32))
    print(bb)





























