# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene data that MDP2 terms read and write, from the articulations of an interactive scene."""

from __future__ import annotations

from isaaclab.scene import InteractiveScene
from isaaclab.utils.warp import ProxyArray

from isaaclab_experimental.mdp2 import EntitySchema, FieldSchema, SceneSchema


class InteractiveSceneData:
    """Live views of the articulation data of an :class:`InteractiveScene`, by scene entity name and field.

    Each articulation provides ``joint_pos``, ``joint_vel``, ``default_joint_pos``, ``default_joint_vel``, and
    ``joint_effort_target``, of shape ``(num_envs, num_joints)``, with its joint and body names. The joint state
    views alias the solver state, so writes reach the solver once the environment commits them.
    """

    def __init__(self, scene: InteractiveScene, step_dt: float):
        entities = {}
        self._views: dict[str, dict[str, ProxyArray]] = {}
        for name, articulation in scene.articulations.items():
            if articulation.cfg.joint_ordering is not None:
                raise ValueError(f"MDP2 needs the solver's joint order, but '{name}' sets a joint ordering.")
            data = articulation.data
            views = {
                "joint_pos": data.joint_pos,
                "joint_vel": data.joint_vel,
                "default_joint_pos": data.default_joint_pos,
                "default_joint_vel": data.default_joint_vel,
                "joint_effort_target": articulation.actuators.target_command.effort,
            }
            entities[name] = EntitySchema(
                names={"joints": tuple(articulation.joint_names), "bodies": tuple(articulation.body_names)},
                fields=dict.fromkeys(views, FieldSchema(("joints",))),
            )
            self._views[name] = views
        self.schema = SceneSchema(num_worlds=scene.num_envs, step_dt=step_dt, entities=entities)

    def data(self, entity: str, field: str) -> ProxyArray:
        """Return a live view of an entity field, of shape ``(num_envs, ...)``."""
        return self._views[entity][field]
