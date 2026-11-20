# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical randomization and normalized context for the native Newton task."""

import torch
import warp as wp
from newton import ModelFlags


def randomize_context(env, env_ids) -> None:
    """Randomize foot friction, torso CoM, and encoder bias for the requested environments."""
    command = env.command_manager.get_term("motion")
    robot = command.robot
    ids = torch.arange(env.num_envs, device=env.device)
    if env_ids is not None:
        ids = ids[env_ids]
    command.friction[ids] = 1.0
    command.encoder_bias[ids] = 0.0
    command.com[ids] = command.nominal_com[ids]
    if not command.cfg.evaluation:
        command.friction[ids] = torch.empty_like(command.friction[ids]).uniform_(0.3, 1.2)
        command.encoder_bias[ids] = torch.empty_like(command.encoder_bias[ids]).uniform_(-0.01, 0.01)
        offset = torch.empty_like(command.com[ids]).uniform_(-1, 1) * command.com.new_tensor((0.025, 0.05, 0.05))
        command.com[ids] += offset
    torso_id = robot.body_names.index(command.cfg.context_body)
    robot.set_coms_index(coms=command.com[ids].unsqueeze(1), body_ids=[torso_id], env_ids=ids)
    physics = env.sim.backend_manager
    model = physics.backend.model
    view = robot.root_view
    values = wp.to_torch(view.get_attribute("shape_material_mu", model)).clone()
    foot_ids = robot.find_bodies(command.cfg.friction_bodies)[0]
    if not foot_ids:
        raise ValueError("Foot friction requires at least one matching body.")
    shape_ids = [shape for body in robot.map_body_ids_to_backend(foot_ids) for shape in view.body_shapes[body]]
    shapes = torch.tensor(shape_ids, device=env.device)
    values[ids[:, None], 0, shapes[None, :]] = command.friction[ids]
    view.set_attribute("shape_material_mu", model, wp.from_torch(values))
    physics.add_model_change(ModelFlags.SHAPE_PROPERTIES)


class SimContext(torch.nn.Module):
    """Reference column order: foot friction, absolute body CoM, encoder bias.

    Normalization uses randomization range midpoint and half-width, as in the
    reference implementation. The CoM channels contain the actual local CoM,
    including its nominal offset, rather than just the sampled perturbation.
    """

    def __init__(self, num_joints: int, device: str):
        super().__init__()
        self.output_dim = 4 + num_joints
        self.register_buffer("mean", torch.tensor([0.75, 0, 0, 0] + [0] * num_joints, device=device))
        self.register_buffer("scale", torch.tensor([0.45, 0.025, 0.05, 0.05] + [0.01] * num_joints, device=device))

    def read(self, env) -> torch.Tensor:
        command = env.command_manager.get_term("motion")
        return (torch.cat((command.friction, command.com, command.encoder_bias), -1) - self.mean) / self.scale

    def find(self, name: str) -> slice | None:
        return {"foot_friction": slice(0, 1), "base_com": slice(1, 4), "encoder_bias": slice(4, self.output_dim)}.get(
            name
        )

    def denormalize(self, sim_ctx: torch.Tensor, columns: slice) -> torch.Tensor:
        return sim_ctx[..., columns] * self.scale[columns] + self.mean[columns]
