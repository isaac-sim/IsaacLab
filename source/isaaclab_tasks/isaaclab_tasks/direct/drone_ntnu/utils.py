import torch

@torch.jit.script
def torch_rand_float_tensor(lower, upper):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    return (upper - lower) * torch.rand_like(upper) + lower
