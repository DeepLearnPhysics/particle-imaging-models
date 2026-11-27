"""
Checkpoint Utils for Models

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import contextlib
import weakref

import torch
import torch.utils.checkpoint


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class _RecomputedTensor:
    """Saved-tensor hooks that store ``input`` in place of ``tensor = fn(input)`` and recompute it when unpacked."""

    def __init__(self, tensor, fn, input):
        self.tensor = weakref.ref(tensor)  # weak, so the tensor itself can be freed
        self.fn = fn
        self.input = input

    def pack(self, saved):
        return self if saved is self.tensor() else saved

    def unpack(self, saved):
        return self.fn(self.input) if saved is self else saved


def recompute_activation(act, hidden, *layers):
    """``layers``, in order, applied to ``act(hidden)`` without keeping ``act(hidden)`` alive until backward.

    For a cheap deterministic ``act`` whose input is saved anyway, such as a GELU, the backward recomputes the
    activation from ``hidden`` instead; outputs and gradients are unchanged and no layer is recomputed. Eager code
    swaps the saved activation through saved-tensor hooks; inside compiled regions the activation is checkpointed,
    so Inductor recomputes it rather than saving it.
    """
    context = contextlib.nullcontext()
    if not torch.is_grad_enabled():
        x = act(hidden)
    elif torch.compiler.is_compiling():
        x = torch.utils.checkpoint.checkpoint(act, hidden, use_reentrant=False)
    else:
        x = act(hidden)
        hooks = _RecomputedTensor(x, act, hidden)
        context = torch.autograd.graph.saved_tensors_hooks(hooks.pack, hooks.unpack)
    with context:
        for layer in layers:
            x = layer(x)
    return x
