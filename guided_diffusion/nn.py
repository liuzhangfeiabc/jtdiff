import jittor as jt
from jittor import init
import math
from jittor import nn

class SiLU(nn.Module):

    def execute(self, x):
        return (x * jt.sigmoid(x))

class GroupNorm32(nn.GroupNorm):

    def execute(self, x):
        # print("GroupNorm32")
        # return super().execute(jt.float32(x)).reshape(x.shape).unary(op=x.dtype)
        return super().execute(x.float()).astype(x.dtype)

def conv_nd(dims, *args, **kwargs):
    if (dims == 1):
        return nn.Conv1d(*args, **kwargs)
    elif (dims == 2):
        return nn.Conv2d(*args, **kwargs)
    elif (dims == 3):
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f'unsupported dimensions: {dims}')

def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)

def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

def scale_module(module, scale):
    for p in module.parameters():
        p.detach().mul_(scale)
    return module

def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def normalization(channels):
    return GroupNorm32(32, channels)

def timestep_embedding(timesteps, dim, max_period=10000):
    half = (dim // 2)
    freqs = jt.exp((((- math.log(max_period)) * jt.arange(start=0, end=half, dtype=jt.float32)) / half))#.to(device=timesteps.device)
    args = (timesteps[:, None].float() * freqs[None])
    embedding = jt.contrib.concat([jt.cos(args), jt.sin(args)], dim=(- 1))
    if (dim % 2):
        embedding = jt.contrib.concat([embedding, jt.zeros_like(embedding[:, :1])], dim=(- 1))
    return embedding

def checkpoint(func, inputs, params, flag):
    if flag:
        args = (tuple(inputs) + tuple(params))
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)

class CheckpointFunction(jt.Function):

    # @staticmethod
    def execute(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with jt.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    # @staticmethod
    def grad(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with jt.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = jt.grad(output_tensors, (ctx.input_tensors + ctx.input_params), output_grads, allow_unused=True)
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return ((None, None) + input_grads)

