# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
if not AppLauncher.instance():
    simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import torch
import pytest

import isaaclab.utils.noise as noise


def test_gaussian_noise():
    """Test guassian_noise function."""

    for device in ["cpu", "cuda"]:
        for noise_device in ["cpu", "cuda"]:
            for op in ["add", "scale", "abs"]:
                # create random data set
                data = torch.rand(10000, 3, device=device)
                # define standard deviation and mean
                std = torch.tensor([0.1, 0.2, 0.3], device=noise_device)
                mean = torch.tensor([0.4, 0.5, 0.6], device=noise_device)
                # create noise config
                noise_cfg = noise.GaussianNoiseCfg(std=std, mean=mean, operation=op)

                for i in range(10):
                    # apply noise
                    noisy_data = noise_cfg.func(data, cfg=noise_cfg)
                    # calculate resulting noise compared to original data set
                    if op == "add":
                        std_result, mean_result = torch.std_mean(noisy_data - data, dim=0)
                    elif op == "scale":
                        std_result, mean_result = torch.std_mean(noisy_data / data, dim=0)
                    elif op == "abs":
                        std_result, mean_result = torch.std_mean(noisy_data, dim=0)

                    assert noise_cfg.mean.device == device
                    assert noise_cfg.std.device == device
                    torch.testing.assert_close(noise_cfg.std, std_result, atol=1e-2, rtol=1e-2)
                    torch.testing.assert_close(noise_cfg.mean, mean_result, atol=1e-2, rtol=1e-2)


def test_uniform_noise():
    """Test uniform_noise function."""
    for device in ["cpu", "cuda"]:
        for noise_device in ["cpu", "cuda"]:
            for op in ["add", "scale", "abs"]:
                # create random data set
                data = torch.rand(10000, 3, device=device)
                # define uniform minimum and maximum
                n_min = torch.tensor([0.1, 0.2, 0.3], device=noise_device)
                n_max = torch.tensor([0.4, 0.5, 0.6], device=noise_device)
                # create noise config
                noise_cfg = noise.UniformNoiseCfg(n_max=n_max, n_min=n_min, operation=op)

                for i in range(10):
                    # apply noise
                    noisy_data = noise_cfg.func(data, cfg=noise_cfg)
                    # calculate resulting noise compared to original data set
                    if op == "add":
                        min_result, _ = torch.min(noisy_data - data, dim=0)
                        max_result, _ = torch.max(noisy_data - data, dim=0)
                    elif op == "scale":
                        min_result, _ = torch.min(torch.div(noisy_data, data), dim=0)
                        max_result, _ = torch.max(torch.div(noisy_data, data), dim=0)
                    elif op == "abs":
                        min_result, _ = torch.min(noisy_data, dim=0)
                        max_result, _ = torch.max(noisy_data, dim=0)

                    assert noise_cfg.n_min.device == device
                    assert noise_cfg.n_max.device == device
                    # add a small epsilon to accommodate for floating point error
                    assert all(torch.le(noise_cfg.n_min - 1e-5, min_result).tolist())
                    assert all(torch.ge(noise_cfg.n_max + 1e-5, max_result).tolist())


def test_constant_noise():
    """Test constant_noise"""
    for device in ["cpu", "cuda"]:
        for noise_device in ["cpu", "cuda"]:
            for op in ["add", "scale", "abs"]:
                # create random data set
                data = torch.rand(10000, 3, device=device)
                # define a bias
                bias = torch.tensor([0.1, 0.2, 0.3], device=noise_device)
                # create noise config
                noise_cfg = noise.ConstantNoiseCfg(bias=bias, operation=op)

                for i in range(10):
                    # apply noise
                    noisy_data = noise_cfg.func(data, cfg=noise_cfg)
                    # calculate resulting noise compared to original data set
                    if op == "add":
                        bias_result = noisy_data - data
                    elif op == "scale":
                        bias_result = noisy_data / data
                    elif op == "abs":
                        bias_result = noisy_data

                    assert noise_cfg.bias.device == device
                    torch.testing.assert_close(noise_cfg.bias.repeat(data.shape[0], 1), bias_result)


if __name__ == "__main__":
    run_tests()
