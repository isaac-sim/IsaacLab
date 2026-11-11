# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene data that MDP2 terms read and write, from the assets of an interactive scene."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

import warp as wp
from newton import ModelFlags

from isaaclab.scene import InteractiveScene
from isaaclab.sensors import BaseContactSensor
from isaaclab.sim import SimulationContext
from isaaclab.utils.warp import ProxyArray

from isaaclab_experimental.mdp2 import EntitySchema, FieldSchema, FieldSelector, SceneSchema


class InteractiveSceneData:
    """Live views of the asset data of an :class:`InteractiveScene`, by scene entity name and field.

    Fields keep the names of the asset data. Articulations and rigid objects provide their root state:

    - ``root_link_pose_w``, ``default_root_pose``: ``wp.transformf`` per world;
    - ``root_com_vel_w``, ``default_root_vel``: ``wp.spatial_vectorf`` per world;
    - ``body_link_pose_w``: ``wp.transformf`` per world and body.

    Articulations also provide ``joint_pos``, ``joint_vel``, ``joint_acc``, ``default_joint_pos``,
    ``default_joint_vel``, ``joint_pos_target``, ``joint_effort_target``, ``applied_effort``,
    ``soft_joint_vel_limits``, and ``soft_joint_pos_limits`` (``wp.vec2f``), per world and joint. The ``scene``
    entity provides ``env_origins`` and ``gravity_w`` (``wp.vec3f``, with magnitude) per world.

    Contact sensors provide ``net_forces_w`` and their histories ``net_forces_w_history`` and
    ``net_normal_forces_w_history`` (``wp.vec3f``, most recent first), and the air and contact times
    ``current_air_time``, ``last_air_time``, ``current_contact_time``, and ``last_contact_time``, per world and body.
    Sensors with filter objects also provide ``normal_force_matrix_w`` (``wp.vec3f``) per world, body, and filter.

    The joint state and the root state of floating bases alias the solver state, so events may write them in place;
    :meth:`commit` makes the solver see the writes. Root writes to fixed bases do not reach the solver. ``gravity_w``
    is the gravity of the physics model, which events may also write.
    """

    def __init__(self, scene: InteractiveScene, step_dt: float):
        self.scene = scene
        assets = {**scene.articulations, **scene.rigid_objects}
        entities: dict[str, EntitySchema] = {}
        self._views: dict[str, dict[str, ProxyArray]] = {}

        # Articulations and rigid objects.
        for name, asset in assets.items():
            data = asset.data
            views = {
                "root_link_pose_w": data.root_link_pose_w,
                "root_com_vel_w": data.root_com_vel_w,
                "default_root_pose": data.default_root_pose,
                "default_root_vel": data.default_root_vel,
                "body_link_pose_w": data.body_link_pose_w,
            }
            fields = {
                "root_link_pose_w": FieldSchema((), wp.transformf),
                "root_com_vel_w": FieldSchema((), wp.spatial_vectorf),
                "default_root_pose": FieldSchema((), wp.transformf),
                "default_root_vel": FieldSchema((), wp.spatial_vectorf),
                "body_link_pose_w": FieldSchema(("bodies",), wp.transformf),
            }
            names = {"bodies": tuple(asset.body_names)}

            if name in scene.articulations:
                if asset.cfg.joint_ordering is not None:
                    raise ValueError(f"MDP2 needs the solver's joint order, but '{name}' sets a joint ordering.")
                views |= {
                    "joint_pos": data.joint_pos,
                    "joint_vel": data.joint_vel,
                    "default_joint_pos": data.default_joint_pos,
                    "default_joint_vel": data.default_joint_vel,
                    "joint_acc": data.joint_acc,
                    "joint_pos_target": asset.actuators.target_command.position,
                    "joint_effort_target": asset.actuators.target_command.effort,
                    "applied_effort": asset.actuators.applied_effort,
                    "soft_joint_pos_limits": data.soft_joint_pos_limits,
                    "soft_joint_vel_limits": data.soft_joint_vel_limits,
                }
                fields |= dict.fromkeys(
                    (
                        "joint_pos",
                        "joint_vel",
                        "joint_acc",
                        "default_joint_pos",
                        "default_joint_vel",
                        "joint_pos_target",
                        "joint_effort_target",
                        "applied_effort",
                        "soft_joint_vel_limits",
                    ),
                    FieldSchema(("joints",)),
                )
                fields["soft_joint_pos_limits"] = FieldSchema(("joints",), wp.vec2f)
                names["joints"] = tuple(asset.joint_names)

            entities[name] = EntitySchema(names=names, fields=fields)
            self._views[name] = views

        # Contact sensors.
        for name, sensor in scene.sensors.items():
            if not isinstance(sensor, BaseContactSensor):
                continue

            data = sensor.data
            forces = {
                "net_forces_w": data.net_forces_w,
                "net_forces_w_history": data.net_forces_w_history,
                "net_normal_forces_w_history": data.net_normal_forces_w_history,
                "normal_force_matrix_w": data.normal_force_matrix_w,
            }
            times = {
                "current_air_time": data.current_air_time,
                "last_air_time": data.last_air_time,
                "current_contact_time": data.current_contact_time,
                "last_contact_time": data.last_contact_time,
            }
            fields = {
                "net_forces_w": FieldSchema(("bodies",), wp.vec3f),
                "net_forces_w_history": FieldSchema(("history", "bodies"), wp.vec3f),
                "net_normal_forces_w_history": FieldSchema(("history", "bodies"), wp.vec3f),
                "normal_force_matrix_w": FieldSchema(("bodies", "filters"), wp.vec3f),
                **dict.fromkeys(times, FieldSchema(("bodies",))),
            }

            # Fields the sensor does not track, like air times without ``track_air_time``, or the forces against
            # filter objects without ``filter_prim_paths_expr``, are left out.
            views = {field: view for field, view in {**forces, **times}.items() if view is not None}
            history = tuple(str(step) for step in range(data.net_forces_w_history.shape[1]))
            matrix = data.normal_force_matrix_w
            filters = () if matrix is None else tuple(str(index) for index in range(matrix.shape[2]))
            entities[name] = EntitySchema(
                names={"bodies": tuple(sensor.body_names), "history": history, "filters": filters},
                fields={field: fields[field] for field in views},
            )
            self._views[name] = views

        # Scene-wide fields.
        self._views["scene"] = {
            "env_origins": ProxyArray(wp.from_torch(scene.env_origins.contiguous(), dtype=wp.vec3f))
        }
        if assets:
            # The data of every asset holds the gravity of every world.
            self._views["scene"]["gravity_w"] = next(iter(assets.values())).data.GRAVITY_VEC_W
        entities["scene"] = EntitySchema(
            names={}, fields=dict.fromkeys(self._views["scene"], FieldSchema((), wp.vec3f))
        )

        self.schema = SceneSchema(num_worlds=scene.num_envs, step_dt=step_dt, entities=entities)

    def data(self, entity: str, field: str) -> ProxyArray:
        """Return a live view of an entity field, of shape ``(num_envs, ...)``."""
        return self._views[entity][field]

    def entity(self, name: str) -> Any:
        """Return the scene object of an entity, such as an articulation."""
        return self.scene[name]

    def commit(self, mask: wp.array, fields: Collection[FieldSelector]) -> None:
        """Recompute the body poses of the masked worlds from their joint and root state, and pass a written gravity
        to the solver. Safe to capture in a CUDA graph."""
        manager = SimulationContext.instance().backend_manager
        if FieldSelector("scene", "gravity_w") in fields:
            manager.add_model_change(ModelFlags.MODEL_PROPERTIES)

        for asset in (*self.scene.articulations.values(), *self.scene.rigid_objects.values()):
            manager.invalidate_fk(env_mask=mask, articulation_ids=asset.root_view.articulation_ids)
        SimulationContext.instance().forward()
