import jittor as jt
from jittor import init
from abc import abstractmethod
import math
import numpy as np
from jittor import nn
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import checkpoint, conv_nd, linear, avg_pool_nd, zero_module, normalization, timestep_embedding

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads_channels: int, output_dim: int=None):
        super().__init__()
        self.positional_embedding = jt.array((jt.randn(embed_dim, ((spacial_dim ** 2) + 1)) / (embed_dim ** 0.5)))
        self.qkv_proj = conv_nd(1, embed_dim, (3 * embed_dim), 1)
        self.c_proj = conv_nd(1, embed_dim, (output_dim or embed_dim), 1)
        self.num_heads = (embed_dim // num_heads_channels)
        self.attention = QKVAttention(self.num_heads)

    def execute(self, x):
        (b, c, *_spatial) = x.shape
        x = x.reshape(b, c, (- 1))
        x = jt.contrib.concat([x.mean(dim=(- 1), keepdim=True), x], dim=(- 1))
        x = (x + self.positional_embedding[None, :, :].to(x.dtype))
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]

class TimestepBlock(nn.Module):
    # '\n    Any module where forward() takes timestep embeddings as a second argument.\n    '

    @abstractmethod
    def execute(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    # '\n    A sequential module that passes timestep embeddings to the children that\n    support it as an extra input.\n    '

    def execute(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    # '\n    An upsampling layer with an optional convolution.\n\n    :param channels: channels in the inputs and outputs.\n    :param use_conv: a bool determining if a convolution is applied.\n    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then\n                 upsampling occurs in the inner-two dimensions.\n    '

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = (out_channels or channels)
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def execute(self, x):
        assert (x.shape[1] == self.channels)
        if (self.dims == 3):
            x = nn.interpolate(x, (x.shape[2], (x.shape[3] * 2), (x.shape[4] * 2)), mode='nearest')
        else:
            x = nn.interpolate(x, scale_factor=2, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    # '\n    A downsampling layer with an optional convolution.\n\n    :param channels: channels in the inputs and outputs.\n    :param use_conv: a bool determining if a convolution is applied.\n    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then\n                 downsampling occurs in the inner-two dimensions.\n    '

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = (out_channels or channels)
        self.use_conv = use_conv
        self.dims = dims
        stride = (2 if (dims != 3) else (1, 2, 2))
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert (self.channels == self.out_channels)
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def execute(self, x):
        assert (x.shape[1] == self.channels)
        return self.op(x)

class ResBlock(TimestepBlock):
    # '\n    A residual block that can optionally change the number of channels.\n\n    :param channels: the number of input channels.\n    :param emb_channels: the number of timestep embedding channels.\n    :param dropout: the rate of dropout.\n    :param out_channels: if specified, the number of out channels.\n    :param use_conv: if True and out_channels is specified, use a spatial\n        convolution instead of a smaller 1x1 convolution to change the\n        channels in the skip connection.\n    :param dims: determines if the signal is 1D, 2D, or 3D.\n    :param use_checkpoint: if True, use gradient checkpointing on this module.\n    :param up: if True, use this block for upsampling.\n    :param down: if True, use this block for downsampling.\n    '

    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_conv=False, use_scale_shift_norm=False, dims=2, use_checkpoint=False, up=False, down=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = (out_channels or channels)
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        # self.in_layers0=nn.Sequential(normalization(channels), Silu())
        # self.in_layers1=conv_nd(dims, channels, self.out_channels, 3, padding=1)
        self.in_layers = nn.Sequential(normalization(channels), Silu(), conv_nd(dims, channels, self.out_channels, 3, padding=1))
        self.updown = (up or down)
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        self.emb_layers = nn.Sequential(Silu(), linear(emb_channels, ((2 * self.out_channels) if use_scale_shift_norm else self.out_channels)))
        # self.out_layers0=normalization(self.out_channels)
        # self.out_layers1=nn.Sequential(Silu(), nn.Dropout(p=dropout), zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)))
        self.out_layers = nn.Sequential(normalization(self.out_channels), Silu(), nn.Dropout(p=dropout), zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)))
        if (self.out_channels == channels):
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def execute(self, x, emb):
        # '\n        Apply the block to a Tensor, conditioned on a timestep embedding.\n\n        :param x: an [N x C x ...] Tensor of features.\n        :param emb: an [N x emb_channels] Tensor of timestep embeddings.\n        :return: an [N x C x ...] Tensor of outputs.\n        '
        # return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)
        return self._forward(x,emb)

    def _forward(self, x, emb):
        # print("execute ResBlock:")
        # print("updown: ", self.updown)
        # print("use_scale_shift_norm: ", self.use_scale_shift_norm)
        # print("x: ", x)
        # print("x.shape: ", x.shape)
        # print("emb: ", emb)
        # print("emb.shape: ", emb.shape)
        # print("in_layer: ", self.in_layers)
        if self.updown:
            # (in_rest, in_conv) = (self.in_layers0, self.in_layers1)
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            # h = in_rest(x)
            h=x
            for ly in in_rest:
                h=ly(h)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
            # h = self.in_layers0(x)
            # h = self.in_layers1(h)
        emb_out = self.emb_layers(emb).astype(h.dtype)
        while (len(emb_out.shape) < len(h.shape)):
            emb_out = emb_out[(..., None)]
        # print("chkpoint 1: ")
        # print("x: ", x)
        # print("in_layer[0]: ", self.in_layers[0])
        # print("in_layer[0].parameter: ", self.in_layers[0].parameters())
        # print("in_layer[0](x): ", self.in_layers[0](x))
        # print("in_layer[0].parameter again: ", self.in_layers[0].parameters())
        # print("in_layer[0](x) again: ", self.in_layers[0](x))
        # print("h: ", h)
        # print("h.shape: ", h.shape)
        # print("emb_out: ", emb_out)
        # print("emb_out.shape: ", emb_out.shape)

        if self.use_scale_shift_norm:
            # (out_norm, out_rest) = (self.out_layers0, self.out_layers1)
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            (scale, shift) = jt.chunk(emb_out, 2, dim=1)
            h = out_norm(h)
            h = ((h * (1 + scale)) + shift)
            # h = out_rest(h)
            for ly in out_rest:
                h=ly(h)
        else:
            h = (h + emb_out)
            h = self.out_layers(h)
            # h = self.out_layers0(h)
            # h = self.out_layers1(h)
        return (self.skip_connection(x) + h)

class AttentionBlock(nn.Module):
    # '\n    An attention block that allows spatial positions to attend to each other.\n\n    Originally ported from here, but adapted to the N-d case.\n    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.\n    '

    def __init__(self, channels, num_heads=1, num_head_channels=(- 1), use_checkpoint=False, use_new_attention_order=False):
        super().__init__()
        self.channels = channels
        if (num_head_channels == (- 1)):
            self.num_heads = num_heads
        else:
            assert ((channels % num_head_channels) == 0), f'q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}'
            self.num_heads = (channels // num_head_channels)
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, (channels * 3), 1)
        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)
        else:
            self.attention = QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def execute(self, x):
        # return checkpoint(self._forward, (x,), self.parameters(), True)
        return self._forward(x)

    def _forward(self, x):
        (b, c, *spatial) = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

def count_flops_attn(model, _x, y):
    # '\n    A counter for the `thop` package to count the operations in an\n    attention operation.\n    Meant to be used like:\n        macs, params = thop.profile(\n            model,\n            inputs=(inputs, timestamps),\n            custom_ops={QKVAttention: QKVAttention.count_flops},\n        )\n    '
    (b, c, *spatial) = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = (((2 * b) * (num_spatial ** 2)) * c)
    model.total_ops += jt.DoubleTensor([matmul_ops])

class QKVAttentionLegacy(nn.Module):
    # '\n    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping\n    '

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def execute(self, qkv):
        # '\n        Apply QKV attention.\n\n        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.\n        :return: an [N x (H * C) x T] tensor after attention.\n        '
        (bs, width, length) = qkv.shape
        assert ((width % (3 * self.n_heads)) == 0)
        ch = (width // (3 * self.n_heads))
        (q, k, v) = qkv.reshape((bs * self.n_heads), (ch * 3), length).split(ch, dim=1)
        scale = (1 / math.sqrt(math.sqrt(ch)))
        weight = jt.einsum('bct,bcs->bts', (q * scale), (k * scale))
        weight = nn.softmax(weight.float(), dim=(- 1)).astype(weight.dtype)
        a = jt.einsum('bts,bcs->bct', weight, v)
        return a.reshape(bs, (- 1), length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class QKVAttention(nn.Module):
    # '\n    A module which performs QKV attention and splits in a different order.\n    '

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def execute(self, qkv):
        # '\n        Apply QKV attention.\n\n        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.\n        :return: an [N x (H * C) x T] tensor after attention.\n        '
        (bs, width, length) = qkv.shape
        assert ((width % (3 * self.n_heads)) == 0)
        ch = (width // (3 * self.n_heads))
        (q, k, v) = qkv.chunk(3, dim=1)
        scale = (1 / math.sqrt(math.sqrt(ch)))
        weight = jt.einsum('bct,bcs->bts', (q * scale).view(((bs * self.n_heads), ch, length)), (k * scale).view(((bs * self.n_heads), ch, length)))
        weight = nn.softmax(weight.float(), dim=(- 1)).astype(weight.dtype)
        a = jt.einsum('bts,bcs->bct', weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, (- 1), length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class UNetModel(nn.Module):
    # '\n    The full UNet model with attention and timestep embedding.\n\n    :param in_channels: channels in the input Tensor.\n    :param model_channels: base channel count for the model.\n    :param out_channels: channels in the output Tensor.\n    :param num_res_blocks: number of residual blocks per downsample.\n    :param attention_resolutions: a collection of downsample rates at which\n        attention will take place. May be a set, list, or tuple.\n        For example, if this contains 4, then at 4x downsampling, attention\n        will be used.\n    :param dropout: the dropout probability.\n    :param channel_mult: channel multiplier for each level of the UNet.\n    :param conv_resample: if True, use learned convolutions for upsampling and\n        downsampling.\n    :param dims: determines if the signal is 1D, 2D, or 3D.\n    :param num_classes: if specified (as an int), then this model will be\n        class-conditional with `num_classes` classes.\n    :param use_checkpoint: use gradient checkpointing to reduce memory usage.\n    :param num_heads: the number of attention heads in each attention layer.\n    :param num_heads_channels: if specified, ignore num_heads and instead use\n                               a fixed channel width per attention head.\n    :param num_heads_upsample: works with num_heads to set a different number\n                               of heads for upsampling. Deprecated.\n    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.\n    :param resblock_updown: use residual blocks for up/downsampling.\n    :param use_new_attention_order: use a different attention pattern for potentially\n                                    increased efficiency.\n    '

    def __init__(self, image_size, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True, dims=2, num_classes=None, use_checkpoint=False, use_fp16=False, num_heads=1, num_head_channels=(- 1), num_heads_upsample=(- 1), use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False):
        super().__init__()
        if (num_heads_upsample == (- 1)):
            num_heads_upsample = num_heads
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = (jt.float16 if use_fp16 else jt.float32)
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        time_embed_dim = (model_channels * 4)
        self.time_embed = nn.Sequential(linear(model_channels, time_embed_dim), Silu(), linear(time_embed_dim, time_embed_dim))
        if (self.num_classes is not None):
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        ch = input_ch = int((channel_mult[0] * model_channels))
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for (level, mult) in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim, dropout, out_channels=int((mult * model_channels)), dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]
                ch = int((mult * model_channels))
                if (ds in attention_resolutions):
                    layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if (level != (len(channel_mult) - 1)):
                out_ch = ch
                self.input_blocks.append(TimestepEmbedSequential((ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True) if resblock_updown else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch))))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
        self.middle_block = TimestepEmbedSequential(ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm), AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order), ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm))
        self._feature_size += ch
        self.output_blocks = nn.ModuleList([])
        for (level, mult) in list(enumerate(channel_mult))[::(- 1)]:
            for i in range((num_res_blocks + 1)):
                ich = input_block_chans.pop()
                layers = [ResBlock((ch + ich), time_embed_dim, dropout, out_channels=int((model_channels * mult)), dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]
                ch = int((model_channels * mult))
                if (ds in attention_resolutions):
                    layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads_upsample, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order))
                if (level and (i == num_res_blocks)):
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
        self.out = nn.Sequential(normalization(ch), Silu(), zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)))

    def convert_to_fp16(self):
        # '\n        Convert the torso of the model to float16.\n        '
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        # '\n        Convert the torso of the model to float32.\n        '
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def execute(self, x, timesteps, y=None):
        '\n        Apply the model to an input batch.\n\n        :param x: an [N x C x ...] Tensor of inputs.\n        :param timesteps: a 1-D batch of timesteps.\n        :param y: an [N] Tensor of labels, if class-conditional.\n        :return: an [N x C x ...] Tensor of outputs.\n        '
        # print("execute: ")
        # print("x: ", x)
        # print("x.shape: ", x.shape)
        # print("timesteps: ", timesteps)
        # print("y: ", y)
        assert ((y is not None) == (self.num_classes is not None)), 'must specify y if and only if the model is class-conditional'
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if (self.num_classes is not None):
            assert (y.shape == (x.shape[0],))
            emb = (emb + self.label_emb(y))
        h = x.astype(self.dtype)
        index = 0
        # print("chkpoint 1:")
        # print("h: ", h)
        # print("h.shape: ", h.shape)
        # print("emb: ", emb)
        # print("emb.shape: ", emb.shape)
        for module in self.input_blocks:
            index += 1
            # print("index begin: ", index)
            h = module(h, emb)
            # if index <= 2:
            #     print("index end: ", index)
            #     print("module: ", module)
            #     print("h: ", h)
            #     print("h.shape: ", h.shape)
            hs.append(h)
        # print("chkpoint 2:")
        # print("h: ", h)
        # print("h.shape: ", h.shape)
        h = self.middle_block(h, emb)

        # print("chkpoint 3:")
        # print("h: ", h)
        # print("h.shape: ", h.shape)
        for module in self.output_blocks:
            h = jt.contrib.concat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.astype(x.dtype)

        # print("chkpoint 4:")
        # print("h: ", h)
        # print("h.shape: ", h.shape)
        return self.out(h)

class SuperResModel(UNetModel):
    '\n    A UNetModel that performs super-resolution.\n\n    Expects an extra kwarg `low_res` to condition on a low-resolution image.\n    '

    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, (in_channels * 2), *args, **kwargs)

    def execute(self, x, timesteps, low_res=None, **kwargs):
        (_, _, new_height, new_width) = x.shape
        upsampled = nn.interpolate(low_res, (new_height, new_width), mode='bilinear')
        x = jt.contrib.concat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)


class Silu(nn.Module):
    def __init__(self):
        '''
        '''

    def execute(self, x):
        x = nn.silu(x)
        return x


class EncoderUNetModel(nn.Module):

    def __init__(self, image_size, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True, dims=2, use_checkpoint=False, use_fp16=False, num_heads=1, num_head_channels=(- 1), num_heads_upsample=(- 1), use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False, pool='adaptive'):
        super().__init__()
        if (num_heads_upsample == (- 1)):
            num_heads_upsample = num_heads
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = (jt.float16 if use_fp16 else jt.float32)
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        time_embed_dim = (model_channels * 4)
        self.time_embed = nn.Sequential(linear(model_channels, time_embed_dim), Silu(), linear(time_embed_dim, time_embed_dim))
        ch = int((channel_mult[0] * model_channels))
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for (level, mult) in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim, dropout, out_channels=int((mult * model_channels)), dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]
                ch = int((mult * model_channels))
                if (ds in attention_resolutions):
                    layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if (level != (len(channel_mult) - 1)):
                out_ch = ch
                self.input_blocks.append(TimestepEmbedSequential((ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True) if resblock_updown else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch))))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
        self.middle_block = TimestepEmbedSequential(ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm), AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order), ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm))
        self._feature_size += ch
        self.pool = pool
        if (pool == 'adaptive'):
            self.out = nn.Sequential(normalization(ch), Silu(), nn.AdaptiveAvgPool2d((1, 1)), zero_module(conv_nd(dims, ch, out_channels, 1)), nn.Flatten())
        elif (pool == 'attention'):
            assert (num_head_channels != (- 1))
            self.out = nn.Sequential(normalization(ch), Silu(), AttentionPool2d((image_size // ds), ch, num_head_channels, out_channels))
        elif (pool == 'spatial'):
            self.out = nn.Sequential(nn.Linear(self._feature_size, 2048), nn.ReLU(), nn.Linear(2048, self.out_channels))
        elif (pool == 'spatial_v2'):
            self.out = nn.Sequential(nn.Linear(self._feature_size, 2048), normalization(2048), Silu(), nn.Linear(2048, self.out_channels))
        else:
            raise NotImplementedError(f'Unexpected {pool} pooling')

    def convert_to_fp16(self):
        '\n        Convert the torso of the model to float16.\n        '
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        '\n        Convert the torso of the model to float32.\n        '
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def execute(self, x, timesteps):
        '\n        Apply the model to an input batch.\n\n        :param x: an [N x C x ...] Tensor of inputs.\n        :param timesteps: a 1-D batch of timesteps.\n        :return: an [N x K] Tensor of outputs.\n        '
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        results = []
        h = x.astype(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith('spatial'):
                results.append(h.astype(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)
        if self.pool.startswith('spatial'):
            results.append(h.astype(x.dtype).mean(dim=(2, 3)))
            h = jt.concat(results, axis=(- 1))
            return self.out(h)
        else:
            h = h.astype(x.dtype)
            return self.out(h)

