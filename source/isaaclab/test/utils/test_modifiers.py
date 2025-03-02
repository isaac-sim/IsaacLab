# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import unittest
from dataclasses import MISSING

import isaaclab.utils.modifiers as modifiers
from isaaclab.utils import configclass


@configclass
class ModifierTestCfg:
    """Configuration for testing modifiers."""

    cfg: modifiers.ModifierCfg = MISSING
    init_data: torch.Tensor = MISSING
    result: torch.Tensor = MISSING
    num_iter: int = 10


class TestModifiers(unittest.TestCase):
    """Test different modifiers implementations."""

    def test_scale_modifier(self):
        """Test for scale modifier."""
        # create a random tensor
        data = torch.rand(128, 128, device="cuda")

        # create a modifier configuration
        cfg = modifiers.ModifierCfg(func=modifiers.scale, params={"multiplier": 2.0})
        # apply the modifier
        processed_data = cfg.func(data, **cfg.params)

        # check if the shape of the modified data is the same as the original data
        self.assertEqual(data.shape, processed_data.shape, msg="Modified data shape does not equal original")
        torch.testing.assert_close(processed_data, data * cfg.params["multiplier"])

    def test_bias_modifier(self):
        """Test for bias modifier."""
        # create a random tensor
        data = torch.rand(128, 128, device="cuda")
        # create a modifier configuration
        cfg = modifiers.ModifierCfg(func=modifiers.bias, params={"value": 0.5})
        # apply the modifier
        processed_data = cfg.func(data, **cfg.params)

        # check if the shape of the modified data is the same as the original data
        self.assertEqual(data.shape, processed_data.shape, msg="Modified data shape does not equal original")
        torch.testing.assert_close(processed_data - data, torch.ones_like(data) * cfg.params["value"])

    def test_clip_modifier(self):
        """Test for clip modifier."""
        # create a random tensor
        data = torch.rand(128, 128, device="cuda")

        # create a modifier configuration
        cfg = modifiers.ModifierCfg(func=modifiers.clip, params={"bounds": (0.5, 2.5)})
        # apply the modifier
        processed_data = cfg.func(data, **cfg.params)

        # check if the shape of the modified data is the same as the original data
        self.assertEqual(data.shape, processed_data.shape, msg="Modified data shape does not equal original")
        self.assertTrue(torch.min(processed_data) >= cfg.params["bounds"][0])
        self.assertTrue(torch.max(processed_data) <= cfg.params["bounds"][1])

    def test_clip_no_upper_bound_modifier(self):
        """Test for clip modifier with no upper bound."""
        # create a random tensor
        data = torch.rand(128, 128, device="cuda")

        # create a modifier configuration
        cfg = modifiers.ModifierCfg(func=modifiers.clip, params={"bounds": (0.0, None)})
        # apply the modifier
        processed_data = cfg.func(data, **cfg.params)

        # check if the shape of the modified data is the same as the original data
        self.assertEqual(data.shape, processed_data.shape, msg="Modified data shape does not equal original")
        self.assertTrue(torch.min(processed_data) >= cfg.params["bounds"][0])

    def test_clip_no_lower_bound_modifier(self):
        """Test for clip modifier with no lower bound."""
        # create a random tensor
        data = torch.rand(128, 128, device="cuda")

        # create a modifier configuration
        cfg = modifiers.ModifierCfg(func=modifiers.clip, params={"bounds": (None, 0.0)})
        # apply the modifier
        processed_data = cfg.func(data, **cfg.params)

        # check if the shape of the modified data is the same as the original data
        self.assertEqual(data.shape, processed_data.shape, msg="Modified data shape does not equal original")
        self.assertTrue(torch.min(processed_data) <= cfg.params["bounds"][1])

    def test_torch_relu_modifier(self):
        """Test for torch relu modifier."""
        # create a random tensor
        data = torch.rand(128, 128, device="cuda")

        # create a modifier configuration
        cfg = modifiers.ModifierCfg(func=torch.nn.functional.relu)
        # apply the modifier
        processed_data = cfg.func(data)

        # check if the shape of the modified data is the same as the original data
        self.assertEqual(data.shape, processed_data.shape, msg="modified data shape does not equal original")
        self.assertTrue(torch.all(processed_data >= 0.0))

    def test_digital_filter(self):
        """Test for digital filter modifier."""
        for device in ["cpu", "cuda"]:
            with self.subTest(device=device):
                # create a modifier configuration
                modifier_cfg = modifiers.DigitalFilterCfg(A=[0.0, 0.1], B=[0.5, 0.5])

                # create a test configuration
                test_cfg = ModifierTestCfg(
                    cfg=modifier_cfg,
                    init_data=torch.tensor([0.0, 0.0, 0.0], device=device).unsqueeze(1),
                    result=torch.tensor([-0.45661893, -0.45661893, -0.45661893], device=device).unsqueeze(1),
                    num_iter=16,
                )

                # create a modifier instance
                modifier_obj = modifier_cfg.func(modifier_cfg, test_cfg.init_data.shape, device=device)

                # test the modifier
                theta = torch.tensor([0.0], device=device)
                delta = torch.pi / torch.tensor([8.0, 8.0, 8.0], device=device).unsqueeze(1)

                for _ in range(5):
                    # reset the modifier
                    modifier_obj.reset()

                    # apply the modifier multiple times
                    for i in range(test_cfg.num_iter):
                        data = torch.sin(theta + i * delta)
                        processed_data = modifier_obj(data)

                        self.assertEqual(
                            data.shape, processed_data.shape, msg="Modified data shape does not equal original"
                        )

                    # check if the modified data is close to the expected result
                    torch.testing.assert_close(processed_data, test_cfg.result)

    def test_integral(self):
        """Test for integral modifier."""
        for device in ["cpu", "cuda"]:
            with self.subTest(device=device):
                # create a modifier configuration
                modifier_cfg = modifiers.IntegratorCfg(dt=1.0)

                # create a test configuration
                test_cfg = ModifierTestCfg(
                    cfg=modifier_cfg,
                    init_data=torch.tensor([0.0], device=device).unsqueeze(1),
                    result=torch.tensor([12.5], device=device).unsqueeze(1),
                    num_iter=6,
                )

                # create a modifier instance
                modifier_obj = modifier_cfg.func(modifier_cfg, test_cfg.init_data.shape, device=device)

                # test the modifier
                delta = torch.tensor(1.0, device=device)

                for _ in range(5):
                    # reset the modifier
                    modifier_obj.reset()

                    # clone the data to avoid modifying the original
                    data = test_cfg.init_data.clone()
                    # apply the modifier multiple times
                    for _ in range(test_cfg.num_iter):
                        processed_data = modifier_obj(data)
                        data = data + delta

                        self.assertEqual(
                            data.shape, processed_data.shape, msg="Modified data shape does not equal original"
                        )

                    # check if the modified data is close to the expected result
                    torch.testing.assert_close(processed_data, test_cfg.result)


if __name__ == "__main__":
    run_tests()
