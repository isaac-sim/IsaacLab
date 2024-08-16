# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from omni.isaac.lab.app import AppLauncher, run_tests

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import unittest
from dataclasses import MISSING

import omni.isaac.lab.utils.modifiers as modifiers
from omni.isaac.lab.utils import configclass


@configclass
class ModifierTestCfg:
    cfg: modifiers.ModifierCfg = MISSING
    init_data: torch.Tensor = MISSING
    result: torch.Tensor = MISSING
    num_iter: int = 10


class TestModifiers(unittest.TestCase):
    """Test different modifiers implementations."""

    def test_scale_modifier(self):
        """Test for Scale modifier."""
        cfg = modifiers.ModifierCfg(func=modifiers.scale, params={"multiplier": 2.0})
        data = torch.ones(3)
        processed_data = cfg.func(data, **cfg.params)
        self.assertEqual(data.shape, processed_data.shape, msg="modified data shape does not equal original")
        self.assertTrue(
            torch.all(torch.isclose(torch.div(processed_data, data), torch.ones_like(data) * cfg.params["multiplier"]))
        )

    def test_bias_modifier(self):
        """Test for Bias Modifier."""
        cfg = modifiers.ModifierCfg(func=modifiers.bias, params={"value": 0.5})
        data = torch.ones(3)
        processed_data = cfg.func(data, **cfg.params)
        self.assertEqual(data.shape, processed_data.shape, msg="modified data shape does not equal original")
        self.assertTrue(torch.all(torch.isclose(processed_data - data, torch.ones_like(data) * cfg.params["value"])))

    def test_clip_modifier(self):
        """Test for Clip Modifier."""
        cfg = modifiers.ModifierCfg(func=modifiers.clip, params={"bounds": (0.5, 2.5)})
        data = torch.ones(3)
        processed_data = cfg.func(data, **cfg.params)
        self.assertEqual(data.shape, processed_data.shape, msg="modified data shape does not equal original")
        self.assertTrue(torch.min(processed_data) >= cfg.params["bounds"][0])
        self.assertTrue(torch.max(processed_data) <= cfg.params["bounds"][1])

    def test_clip_no_upper_bound_modifier(self):
        """Test for Clip Modifier with no upper bound."""
        cfg = modifiers.ModifierCfg(func=modifiers.clip, params={"bounds": (0.0, None)})
        data = torch.ones(3)
        processed_data = cfg.func(data, **cfg.params)
        self.assertEqual(data.shape, processed_data.shape, msg="modified data shape does not equal original")
        self.assertTrue(torch.min(processed_data) >= cfg.params["bounds"][0])

    def test_torch_relu_modifier(self):
        """Test for Torch Relu Modifier."""
        cfg = modifiers.ModifierCfg(func=torch.nn.functional.relu)
        data = torch.rand(128, 128, device="cuda")
        processed_data = cfg.func(data)
        self.assertEqual(data.shape, processed_data.shape, msg="modified data shape does not equal original")
        self.assertTrue(torch.all(processed_data >= 0.0))

    def test_digital_filter(self):
        """Test for Digital Filter modifiers."""

        test_cfg = ModifierTestCfg(
            cfg=modifiers.ModifierCfg(
                func=modifiers.DigitalFilter, params={"A": torch.tensor([0.0, 0.1]), "B": torch.tensor([0.5, 0.5])}
            ),
            init_data=torch.tensor([0.0, 0.0, 0.0]).unsqueeze(1),
            result=torch.tensor([-0.45661893, -0.45661893, -0.45661893]).unsqueeze(1),
            num_iter=16,
        )

        test_cfg.cfg.func = test_cfg.cfg.func(test_cfg.cfg, test_cfg.init_data.shape)
        theta = torch.tensor([0.0])
        delta = torch.pi / torch.tensor([8.0, 8.0, 8.0]).unsqueeze(1)

        for i in range(test_cfg.num_iter):
            data = torch.sin(theta + i * delta)
            processed_data = test_cfg.cfg.func(data)
            self.assertEqual(data.shape, processed_data.shape, msg="modified data shape does not equal original")

        self.assertTrue(torch.all(torch.isclose(processed_data, test_cfg.result)))

    def test_integral(self):
        """Test for Integral Modifier."""
        test_cfg = ModifierTestCfg(
            cfg=modifiers.ModifierCfg(func=modifiers.Integrator, params={"dt": 1.0}),
            init_data=torch.tensor(0.0),
            result=torch.tensor([12.5]),
            num_iter=6,
        )

        data = test_cfg.init_data
        test_cfg.cfg.func = test_cfg.cfg.func(test_cfg.cfg, test_cfg.init_data.shape)
        delta = torch.tensor(1.0)

        for _ in range(test_cfg.num_iter):
            processed_data = test_cfg.cfg.func(data)
            data = data + delta
            self.assertEqual(data.shape, processed_data.shape, msg="modified data shape does not equal original")
        self.assertTrue(torch.all(torch.isclose(processed_data, test_cfg.result)))


if __name__ == "__main__":
    run_tests()
