# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record a task's declared exploration environment into a native NeRD dataset."""

# ruff: noqa: E402
# Configure Warp before importing simulation modules.

import argparse
from pathlib import Path

import warp as wp

wp.config.enable_backward = False

import torch
from isaaclab_neural.physics.nerd_solver import physical_contract
from isaaclab_neural.record.recording import free_acceleration_indices
from newton_neural.backbones import BackboneFlash
from newton_neural.contacts import ContactInputProbes
from newton_neural.data import TrajectoryWriter
from newton_neural.models import FlashNeRD

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import string_to_callable, to_dict
from isaaclab.utils.io import dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--num_frames", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    add_launcher_args(parser)
    args, overrides = setup_preset_cli(parser, argv)
    env_cfg, model_cfg = resolve_task_config(
        args.task,
        "nerd_cfg_entry_point",
        overrides=overrides,
        env_cfg_entry_point="nerd_recording_cfg_entry_point",
    )
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    with launch_simulation(env_cfg, args):
        env = ManagerBasedRLEnv(env_cfg)
        capture = env.recorder_manager.get_term("dynamics")
        try:
            exploration = env.action_manager.get_term("joint_targets")
            backend, binding = env.sim.backend_manager.backend, exploration.binding
            contact = None
            if model_cfg.contact_bodies:
                contact = ContactInputProbes.from_model(
                    backend.model,
                    binding.system,
                    bodies=model_cfg.contact_bodies,
                    articulation=binding.view.articulation_ids.numpy().ravel().tolist(),
                )
            nerd = FlashNeRD(
                binding.system,
                backbone=BackboneFlash(spans=model_cfg.backbone_spans),
                contact_input=contact,
                prediction=string_to_callable(model_cfg.prediction)(),
            ).to(env.device)
            free_motion = not nerd.prediction.integrates
            nerd.metadata["isaaclab"] = {
                "physical_model": physical_contract(backend.model, binding, free_motion=free_motion),
                "model_cfg": to_dict(model_cfg),
            }
            capture.binding, capture.backend = binding, backend
            capture.sensor = None if contact is None else contact.sensor(backend.model, binding)
            capture.free_columns = free_acceleration_indices(backend.solver, binding) if free_motion else None
            capture.writer = TrajectoryWriter(
                args.dataset, nerd.signals, env.step_dt, generation=env.recorder_manager.get_ep_meta()
            )
            action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
            env.reset()
            for _ in range(args.num_frames):
                env.step(action)
            capture.close()
            nerd.save(args.dataset / "model.pt")
            dump_yaml(str(args.dataset / "params" / "env.yaml"), env_cfg)
            dump_yaml(str(args.dataset / "params" / "model.yaml"), model_cfg)
        except BaseException:
            capture.close(complete=False)
            raise
        finally:
            env.close()


if __name__ == "__main__":
    main()
