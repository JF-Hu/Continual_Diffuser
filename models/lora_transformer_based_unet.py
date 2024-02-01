import torch
from torch import nn, einsum
from models.transformer_based_unet import Bernoulli, TimestepBlock, ResBlock, default, CrossAttention, FeedForward, Normalize, exists, SpatialTransformer, AlterableHeadDimTransformer
from models.position_embedding import SinusoidalPosEmb
from models.model_util import normalization, conv_nd, calculate_convolution_parameter_setting
from einops import rearrange, repeat
from models.lora_layer import LoRALinear
from config.hyperparameter import FineTuneParaType


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None, task_name=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer) or isinstance(layer, AlterableHeadDimTransformer):
                x = layer(x, context)
            elif isinstance(layer, SpatialLoRATransformer):
                x = layer(x, task_name, context)
            else:
                x = layer(x)
        return x

class LoRACrossAttention(nn.Module):
    def __init__(self, query_dim, task_names, lora_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = LoRALinear(nn.Linear(context_dim, inner_dim, bias=False), task_name=task_names, r=lora_dim)
        self.to_v = LoRALinear(nn.Linear(context_dim, inner_dim, bias=False), task_name=task_names, r=lora_dim)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, task_names, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context, task_names)
        v = self.to_v(context, task_names)

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

class BasicTransformerLoRABlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, task_names, lora_dim, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = LoRACrossAttention(
            query_dim=dim, task_names=task_names, lora_dim=lora_dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, task_names, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), task_names=task_names, context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

class SpatialLoRATransformer(nn.Module):
    def __init__(self, in_channels, n_heads, d_head, task_names, lora_dim, depth=1, dropout=0., context_dim=None, calc_energy=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        self.norm = Normalize(in_channels)

        self.proj_in = conv_nd(1, in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerLoRABlock(
                dim=inner_dim, n_heads=n_heads, d_head=d_head, task_names=task_names, lora_dim=lora_dim, dropout=dropout, context_dim=context_dim) for d in range(depth)])

        self.proj_out = conv_nd(1, inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, task_names, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h -> b h c')
        for block in self.transformer_blocks:
            x = block(x, task_names=task_names, context=context)
        x = rearrange(x, 'b h c -> b c h', h=h)
        x = self.proj_out(x)
        return x + x_in

class SpatialLoRATransformerUnet(nn.Module):
    def __init__(
            self, input_channels, sequence_length, out_channels, num_res_blocks, attention_resolutions, num_heads,
            model_channels, task_names, lora_dim,
            use_fp16=False, channel_mult=(1, 2, 4, 8), context_dim=None, dropout=0., per_head_channels=-1,
            legacy=True, use_spatial_transformer=True, transformer_depth=1, same_conv_kernel_size=5):
        super().__init__()
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.task_names = task_names
        self.lora_dim = lora_dim
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
        # self.time_mlp = nn.Sequential(
        #     SinusoidalPosEmb(model_channels),
        #     nn.Linear(model_channels, time_embed_dim),
        #     act_fn,
        #     nn.Linear(time_embed_dim, time_embed_dim),
        # )
        self.time_emb = SinusoidalPosEmb(model_channels)
        self.time_mlp1 = LoRALinear(nn.Linear(model_channels, time_embed_dim), task_name=task_names, r=lora_dim)
        self.time_act1 = act_fn
        self.time_mlp2 = LoRALinear(nn.Linear(time_embed_dim, time_embed_dim), task_name=task_names, r=lora_dim)

        context_embed_dim = model_channels
        self.context_mlp1 = LoRALinear(nn.Linear(self.context_dim, model_channels), task_name=task_names, r=lora_dim * 2)
        self.context_act1 = act_fn
        self.context_mlp2 = LoRALinear(nn.Linear(model_channels, context_embed_dim), task_name=task_names, r=lora_dim * 2)
        # self.context_mlp = nn.Sequential(
        #     nn.Linear(self.context_dim, model_channels),
        #     act_fn,
        #     nn.Linear(model_channels, context_embed_dim),
        # )
        self.context_mask_dist = Bernoulli(probs=1 - 0.25)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(conv_nd(1, input_channels, model_channels, self.same_conv_kernel_size,
                                                padding=self.same_conv_kernel_size // 2))
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
                        input_channels=ch, time_emb_channels=time_embed_dim, dropout=dropout,
                        out_channels=mult * model_channels,
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
                    layers.append(SpatialLoRATransformer(
                        in_channels=ch, n_heads=num_heads, d_head=dim_head,
                        task_names=task_names, lora_dim=lora_dim, depth=transformer_depth, context_dim=context_embed_dim))
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
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else per_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                input_channels=ch, time_emb_channels=time_embed_dim, dropout=0, out_channels=ch,
                convolution_type=1, use_checkpoint=False, use_scale_shift_norm=False,
                same_conv_kernel_size=self.same_conv_kernel_size),
            SpatialLoRATransformer(
                in_channels=ch, n_heads=num_heads, d_head=dim_head,
                task_names=task_names, lora_dim=lora_dim, depth=transformer_depth, context_dim=context_embed_dim),
            ResBlock(
                input_channels=ch, time_emb_channels=time_embed_dim, dropout=0, out_channels=ch,
                convolution_type=1, use_checkpoint=False, use_scale_shift_norm=False,
                same_conv_kernel_size=self.same_conv_kernel_size),
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
                        input_channels=ch + ich, time_emb_channels=time_embed_dim, dropout=0,
                        out_channels=model_channels * mult,
                        convolution_type=1, use_checkpoint=False, use_scale_shift_norm=False,
                        same_conv_kernel_size=self.same_conv_kernel_size)
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
                    layers.append(SpatialLoRATransformer(
                        in_channels=ch, n_heads=num_heads, d_head=dim_head,
                        task_names=task_names, lora_dim=lora_dim, depth=transformer_depth, context_dim=context_embed_dim))
                if len(self.convolution_inout_recoder) > 0 and convolution_recoder[-1]["out_length"] != \
                        self.convolution_inout_recoder[-1][-1]["out_length"]:
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

    def forward(self, x, cond, timesteps, task_name, context=None, use_dropout=True, force_dropout=False, **kwargs):
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
        # time_emb = self.time_mlp(timesteps.to(x.device))
        time_emb = self.time_emb(timesteps.to(x.device))
        time_emb = self.time_act1(self.time_mlp1(time_emb, task_name))
        time_emb = self.time_mlp2(time_emb, task_name)

        context_emb = self.context_act1(self.context_mlp1(context.to(x.device), task_name))
        context_emb = self.context_mlp2(context_emb, task_name)
        # context_emb = self.context_mlp(context.to(x.device))
        if use_dropout:
            mask = self.context_mask_dist.sample(sample_shape=(context_emb.size(0), 1)).to(context_emb.device)
            context_emb = mask * context_emb
        if force_dropout:
            context_emb = 0 * context_emb
        context_emb = torch.unsqueeze(context_emb, dim=1)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, time_emb, context_emb, task_name)
            hs.append(h)
        h = self.middle_block(h, time_emb, context_emb, task_name)
        for module in self.output_blocks:
            res_h = hs.pop()
            h = torch.cat([h, res_h], dim=1)
            h = module(h, time_emb, context_emb, task_name)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            h = self.id_predictor(h)
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
                 "out_length": calculate_convolution_parameter_setting(H_in=in_length, H_out=None,
                                                                       kernel_size=each_result["kernel_size"],
                                                                       stride=each_result["stride"],
                                                                       padding=each_result["padding"]),
                 })
            in_length = each_result["out_length"]
            result[i] = each_result
        return result

    def recoder_convolution_info(self, block):
        out = self.calculate_convolution_setting(block=block, in_length=self.in_length)
        self.convolution_inout_recoder.append(out)
        self.in_length = out[-1]["out_length"]

    def generate_LoRA_parameters(self, argus, task_name, list_out=True):
        LoRA_parameters = [] if list_out else {}
        if argus.finetune_para_type in [FineTuneParaType.CrossAttention]:
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





