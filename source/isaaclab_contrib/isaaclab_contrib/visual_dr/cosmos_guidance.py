# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Image-transfer mask guidance for Cosmos checkouts without a mask API.

Projection is applied before each velocity evaluation at the preceding update's
noise level, and after the final update. This matches the GitLab post-update hook
without copying either sampler or patching global Cosmos classes.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch.nn.functional as F


def project_sampling(sample, velocity_fn, noise, *args, source, mask, threshold, steps, train_steps, **kwargs):
    """Run a sampler with source projection on the target vision suffix."""
    states = noise if isinstance(noise, list) else [noise]
    if len(states) != 1:
        raise ValueError("The public Cosmos mask adapter requires singleton image transfer")
    source = source.flatten().to(states[0])
    mask = mask.flatten().to(states[0])
    if source.numel() != mask.numel() or source.numel() > states[0].numel():
        raise ValueError("Source/mask latents do not match the sampler target")
    initial = states[0][-source.numel() :].clone()
    calls = 0

    def project(state, timestep, index):
        if index > threshold:
            return
        tensors = state if isinstance(state, list) else [state]
        sigma = (timestep.reshape(()) / train_steps).clamp(0, 1)
        target = tensors[0][-source.numel() :]
        target.copy_(mask * ((1 - sigma) * source + sigma * initial) + (1 - mask) * target)

    def velocity(state, timestep):
        nonlocal calls
        project(state, timestep, max(0, calls - 1))
        calls += 1
        return velocity_fn(state, timestep)

    result = sample(velocity, noise, *args, **kwargs)
    if calls != steps:
        raise RuntimeError(f"Expected {steps} sampler evaluations, got {calls}")
    project(result, initial.new_zeros(()), steps - 1)
    return result


@contextmanager
def guided_image_sampling(model, source, preserve, *, steps: int, threshold: int, strength: float, mode: str):
    """Temporarily wrap this model's sampler; preserve upstream callbacks and CFG."""
    from cosmos_framework.model.generator.diffusion.samplers.fixed_step import FixedStepSampler
    from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler

    latent = model.encode(source).contiguous()
    if latent.shape[2] != 1:
        raise ValueError("Runtime Cosmos guidance expects a single image")
    size = latent.shape[-2:]
    if mode == "max":
        mask = F.adaptive_max_pool2d(preserve, size)
    elif mode == "area":
        mask = F.interpolate(preserve, size=size, mode="area")
    else:
        mask = F.interpolate(preserve, size=size, mode="bilinear", align_corners=False)
    mask = (mask.unsqueeze(2) * strength).expand_as(latent).contiguous()
    options = dict(
        source=latent,
        mask=mask,
        threshold=threshold,
        steps=steps,
        train_steps=model.config.rectified_flow_inference_config.num_train_timesteps,
    )
    name = "fixed_step_sampler" if model.fixed_step_sampler is not None else "sampler"
    original = getattr(model, name)

    class GuidedFixedStep(FixedStepSampler):
        def __call__(self, velocity_fn, noise, *args, **kwargs):
            return project_sampling(original, velocity_fn, noise, *args, **options, **kwargs)

    class GuidedUniPC(UniPCSampler):
        def forward(self, velocity_fn, noise, *args, **kwargs):
            return project_sampling(original, velocity_fn, noise, *args, **options, **kwargs)

    if isinstance(original, FixedStepSampler):
        wrapped = GuidedFixedStep(original.t_list, original.sample_type, original.num_train_timesteps)
    elif isinstance(original, UniPCSampler):
        wrapped = GuidedUniPC(original.cfg, original.tensor_kwargs)
    else:
        raise ValueError("Mask guidance supports only Cosmos UniPC and fixed-step samplers")
    setattr(model, name, wrapped)
    try:
        yield
    finally:
        setattr(model, name, original)
