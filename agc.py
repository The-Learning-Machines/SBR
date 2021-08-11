#########################################################################
# This code was modified from https://github.com/vballoli/nfnets-pytorch #
#########################################################################

import torch
from collections import Iterable
from torch import nn, optim


def unitwise_norm(x: torch.Tensor):
    if x.ndim <= 1:
        dim = 0
        keepdim = False
    elif x.ndim in [2, 3]:
        dim = 0
        keepdim = True
    elif x.ndim == 4:
        dim = [1, 2, 3]
        keepdim = True
    else:
        raise ValueError('Wrong dimensions of x')

    return torch.sum(x**2, dim=dim, keepdim=keepdim) ** 0.5


class AGC(optim.Optimizer):
    
    """Generic implementation of the Adaptive Gradient Clipping
    Args:
      params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
      optim (torch.optim.Optimizer): Optimizer with base class optim.Optimizer
      clipping (float, optional): clipping value (default: 1e-3)
      eps (float, optional): eps (default: 1e-3)
      model (torch.nn.Module, optional): The original model
      ignore_agc (str, Iterable, optional): Layers for AGC to ignore
    """

    def __init__(self, params, optim: optim.Optimizer, clipping: float = 1e-2, eps: float = 1e-3, model=None, ignore_agc= ['']):
        if clipping < 0.0:
            raise ValueError("Invalid clipping value: {}".format(clipping))
        if eps < 0.0:
            raise ValueError("Invalid eps value: {}".format(eps))

        self.optim = optim

        defaults = dict(clipping=clipping, eps=eps)
        defaults = {**defaults, **optim.defaults}

        if not isinstance(ignore_agc, Iterable):
            ignore_agc = [ignore_agc]

        if model is not None:
            assert ignore_agc not in [
                None, []], "Specify args ignore_agc to ignore fc-like(or other) layers"
            names = [name for name, module in model.named_modules()]

            for module_name in ignore_agc:
                if module_name not in names:
                    raise ModuleNotFoundError(
                        "Module name {} not found in the model".format(module_name))
            parameters = [{"params": module.parameters()} for name,
                          module in model.named_modules() if name not in ignore_agc]

        super(AGC, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_norm = torch.max(unitwise_norm(
                    p.detach()), torch.tensor(group['eps']).to(p.device))
                grad_norm = unitwise_norm(p.grad.detach())
                max_norm = param_norm * group['clipping']

                trigger = grad_norm < max_norm

                clipped_grad = p.grad * \
                    (max_norm / torch.max(grad_norm,
                                          torch.tensor(1e-6).to(grad_norm.device)))
                p.grad.data.copy_(torch.where(trigger, clipped_grad, p.grad))

        return self.optim.step(closure)