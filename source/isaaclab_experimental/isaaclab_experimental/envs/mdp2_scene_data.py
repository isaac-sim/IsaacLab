# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene data that MDP2 terms read and write, from the assets of an interactive scene."""

from __future__ import annotations

import functools
from collections.abc import Callable, Collection
from typing import Any

import warp as wp

from isaaclab.actuators import (
    ActuatorBase,
    ActuatorNetLSTMCfg,
    ActuatorNetMLPCfg,
    DCMotorCfg,
    IdealPDActuatorCfg,
    ImplicitActuatorCfg,
)
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import BaseContactSensor, BaseJointWrenchSensor
from isaaclab.sim import SimulationContext
from isaaclab.utils.warp import ProxyArray

from isaaclab_experimental.mdp2 import EntitySchema, FieldSchema, FieldSelector, SceneSchema

CAPTURED_ACTUATORS = (ImplicitActuatorCfg, IdealPDActuatorCfg, DCMotorCfg)
"""Actuator configs whose actuators MDP2 can step and reset inside CUDA graphs. Matched exactly: subclasses, such as
delayed actuators, may draw from Torch's random generators on reset, which a replayed graph repeats."""

NEWTON_NETWORK_ACTUATORS = (ActuatorNetLSTMCfg, ActuatorNetMLPCfg)
"""Actuator network configs that MDP2 supports when Newton runs them as its own neural drives
(``use_newton_actuators``), not as Isaac Lab's Torch models. Newton resets them. Newton cannot capture networks loaded
from TorchScript checkpoints, so for them the environment runs the physics and the scene reset between its graphs."""


class InteractiveSceneData:
    """Live views of the asset data of an :class:`InteractiveScene`, by scene entity name and field.

    Fields keep the names, units, and frames of the asset data: positions in m, orientations as ``(x, y, z, w)``
    quaternions, joint values in rad or m for revolute or prismatic joints, and times in s. Articulations and rigid
    objects provide their root state:

    - ``root_link_pose_w``, ``default_root_pose``: ``wp.transformf`` per world, in the world frame and in the
      frame of the world's environment origin respectively;
    - ``root_com_vel_w``, ``default_root_vel``: ``wp.spatial_vectorf`` per world, linear [m/s] then angular
      [rad/s], in the world frame;
    - ``body_link_pose_w``: ``wp.transformf`` per world and body, in the world frame.

    Articulations provide ``body_com_vel_w`` (a world-frame ``wp.spatial_vectorf`` per world and body, linear [m/s]
    then angular [rad/s]). They also provide ``joint_pos``, ``joint_vel``, ``joint_acc`` [rad/s^2 or m/s^2],
    ``default_joint_pos``, ``default_joint_vel``, ``joint_pos_target``, ``joint_effort_target``,
    ``applied_effort`` [N·m or N], ``soft_joint_vel_limits``, and ``soft_joint_pos_limits`` (``wp.vec2f``), per world
    and joint. The ``scene`` entity provides ``env_origins`` [m] and ``gravity_w`` (``wp.vec3f``, with magnitude)
    [m/s^2] per world.

    Contact sensors provide ``net_forces_w`` [N] and their histories ``net_forces_w_history`` and
    ``net_normal_forces_w_history`` (``wp.vec3f``, most recent first), and the air and contact times
    ``current_air_time``, ``last_air_time``, ``current_contact_time``, and ``last_contact_time``, per world and body.
    Sensors with filter objects also provide ``normal_force_matrix_w`` (``wp.vec3f``) [N] per world, body, and
    filter.

    Joint wrench sensors provide ``force`` [N] and ``torque`` [N·m] as ``wp.vec3f`` per world and body, in the
    incoming joint frame.

    Events and actions write fields in place, and :meth:`commit` makes the simulation see the writes to the joint and
    root state. Where the asset's data is the solver's own state, as Newton's joint positions and root state usually
    are, a commit only tells the solver which worlds changed. Otherwise, as for the joint positions of Newton
    articulations with ball joints or on other physics backends, it passes the written worlds through the asset's
    ``write_*_to_sim_mask`` methods; joint velocities always go through them, since the asset keeps the previous ones
    for joint accelerations. The
    default state and the joint targets are plain buffers that need no commit. ``gravity_w`` is writable where it is
    the physics model's own (Newton). Derived and sensor fields are read-only.
    """

    def __init__(self, scene: InteractiveScene, sim: SimulationContext, step_dt: float) -> None:
        self.scene = scene
        self.sim = sim
        assets = {**scene.articulations, **scene.rigid_objects}
        entities: dict[str, EntitySchema] = {}
        self._views: dict[str, dict[str, ProxyArray]] = {}
        # How a commit passes the writes of each state field to the simulation: through the asset's writer, or, for a
        # field that is the solver's own state, by telling the solver which worlds of which asset changed.
        self._writers: dict[FieldSelector, Callable[[wp.array], None]] = {}
        self._solver_state: dict[FieldSelector, Any] = {}  # The asset whose kinematics a write invalidates
        self._model_changes: dict[FieldSelector, int] = {}  # Model properties that a write changes
        newton = _newton_backend(sim)

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
                "root_link_pose_w": FieldSchema((), wp.transformf, writable=True),
                "root_com_vel_w": FieldSchema((), wp.spatial_vectorf, writable=True),
                "default_root_pose": FieldSchema((), wp.transformf, writable=True),
                "default_root_vel": FieldSchema((), wp.spatial_vectorf, writable=True),
                # Forward kinematics computes it from the root and joint state.
                "body_link_pose_w": FieldSchema(
                    ("bodies",),
                    wp.transformf,
                    derived_from=("root_link_pose_w", "joint_pos")
                    if name in scene.articulations
                    else ("root_link_pose_w",),
                ),
            }
            names = {"bodies": tuple(asset.body_names)}
            state = {
                "root_link_pose_w": lambda mask,
                a=asset,
                v=views["root_link_pose_w"]: a.write_root_link_pose_to_sim_mask(root_pose=v.warp, env_mask=mask),
                "root_com_vel_w": lambda mask, a=asset, v=views["root_com_vel_w"]: (
                    a.write_root_com_velocity_to_sim_mask(root_velocity=v.warp, env_mask=mask)
                ),
            }

            if name in scene.articulations:
                # Terms read the asset's data in kernels, which the asset does not reorder after binding.
                if asset.cfg.joint_ordering is not None or asset.cfg.body_ordering is not None:
                    raise ValueError(
                        f"MDP2 needs the solver's joint and body order, but '{name}' sets a joint or body ordering."
                    )
                # Captured steps replay the actuators' masked resets, so they must not draw from Torch's random
                # generators, as delayed actuators do for their delays: a graph would repeat the same draws. Actuator
                # networks are supported where Newton runs them, not Isaac Lab's Torch models.
                unsupported = sorted(
                    group
                    for group, cfg in asset.cfg.actuators.items()
                    if type(cfg) not in CAPTURED_ACTUATORS
                    and (type(cfg) not in NEWTON_NETWORK_ACTUATORS or isinstance(asset.actuators[group], ActuatorBase))
                )
                if unsupported:
                    raise ValueError(
                        f"MDP2 supports {', '.join(cfg.__name__ for cfg in CAPTURED_ACTUATORS)}, and"
                        f" {', '.join(cfg.__name__ for cfg in NEWTON_NETWORK_ACTUATORS)} that Newton runs, but"
                        f" '{name}' has the actuator groups {unsupported}."
                    )
                views |= {
                    "body_com_vel_w": data.body_com_vel_w,
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
                    ("joint_acc", "applied_effort", "soft_joint_vel_limits"), FieldSchema(("joints",))
                )
                fields |= dict.fromkeys(
                    (
                        "joint_pos",
                        "joint_vel",
                        "default_joint_pos",
                        "default_joint_vel",
                        "joint_pos_target",
                        "joint_effort_target",
                    ),
                    FieldSchema(("joints",), writable=True),
                )
                fields["soft_joint_pos_limits"] = FieldSchema(("joints",), wp.vec2f)
                fields["body_com_vel_w"] = FieldSchema(("bodies",), wp.spatial_vectorf)
                names["joints"] = tuple(asset.joint_names)
                state |= {
                    "joint_pos": lambda mask, a=asset, v=views["joint_pos"]: a.write_joint_position_to_sim_mask(
                        position=v.warp, env_mask=mask
                    ),
                    "joint_vel": lambda mask, a=asset, v=views["joint_vel"], skip=False: (
                        a.write_joint_velocity_to_sim_mask(velocity=v.warp, env_mask=mask, skip_forward=skip)
                    ),
                }

            for field, writer in state.items():
                selector = FieldSelector(name, field)
                if newton is not None and newton.is_solver_state(asset, field):
                    self._solver_state[selector] = asset
                    if (change := newton.model_change(asset, field)) is not None:
                        self._model_changes[selector] = change
                    if newton.keeps_derived(field):
                        # The asset updates what it derives, in its writer's kernel, and the commit recomputes the
                        # kinematics once per asset.
                        self._writers[selector] = functools.partial(writer, skip=True)
                else:
                    self._writers[selector] = writer

            entities[name] = EntitySchema(names=names, fields=fields)
            self._views[name] = views

        for name, sensor in scene.sensors.items():
            if isinstance(sensor, BaseJointWrenchSensor):
                data = sensor.data
                views = {"force": data.force, "torque": data.torque}
                self._views[name] = {field: view for field, view in views.items() if view is not None}
                entities[name] = EntitySchema(
                    names={"bodies": tuple(sensor.body_names)},
                    fields=dict.fromkeys(self._views[name], FieldSchema(("bodies",), wp.vec3f)),
                )
                continue
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
        scene_fields = {"env_origins": FieldSchema((), wp.vec3f)}
        if assets:
            # The data of every asset holds the gravity of every world, which is the model's own on Newton.
            self._views["scene"]["gravity_w"] = next(iter(assets.values())).data.GRAVITY_VEC_W
            scene_fields["gravity_w"] = FieldSchema((), wp.vec3f, writable=newton is not None)
            if newton is not None:
                self._model_changes[FieldSelector("scene", "gravity_w")] = newton.model_properties
        entities["scene"] = EntitySchema(names={}, fields=scene_fields)

        self.schema = SceneSchema(num_worlds=scene.num_envs, step_dt=step_dt, entities=entities)

    def data(self, entity: str, field: str) -> ProxyArray:
        """Return a live view of an entity field, of shape ``(num_envs, ...)``."""
        return self._views[entity][field]

    def entity(self, name: str) -> Any:
        """Return the scene object of an entity, such as an articulation."""
        return self.scene[name]

    def commit(self, mask: wp.array, fields: Collection[FieldSelector]) -> None:
        """Pass the written fields of the masked worlds to the simulation, and recompute their body poses. Safe to
        capture in a CUDA graph.

        Only the assets whose state was written are recomputed, so a commit costs the fields it passes on.
        """
        changed = {id(asset): asset for selector, asset in self._solver_state.items() if selector in fields}
        if changed or any(selector in fields for selector in self._model_changes):
            manager = self.sim.backend_manager
            for change in {self._model_changes[selector] for selector in fields if selector in self._model_changes}:
                manager.add_model_change(change)
            for asset in changed.values():
                manager.invalidate_fk(env_mask=mask, articulation_ids=asset.root_view.articulation_ids)
        for selector in fields:
            if selector in self._writers:
                self._writers[selector](mask)
        self.sim.forward()


class _NewtonBackend:
    """What MDP2 knows about Newton's data, for its fast path: which asset fields are the solver's own state, which
    events then write in place, and which model properties a write changes."""

    def __init__(self) -> None:
        from newton import ModelFlags  # noqa: PLC0415

        self.model_properties = ModelFlags.MODEL_PROPERTIES
        self.joint_properties = ModelFlags.JOINT_PROPERTIES

    def is_solver_state(self, asset: Any, field: str) -> bool:
        """Whether a state field of an asset is the solver's own buffer, so that the commit only tells the solver which
        worlds changed. Joint positions are not for articulations with ball joints, whose quaternion coordinates the
        asset converts."""
        if field != "joint_pos":
            return True
        view = asset.root_view
        return not view.joint_coord_counts or view.joint_coord_counts == view.joint_dof_counts

    @staticmethod
    def keeps_derived(field: str) -> bool:
        """Whether the asset keeps data derived from a field that its writer updates: the previous joint velocities,
        from which it computes joint accelerations."""
        return field == "joint_vel"

    def model_change(self, asset: Any, field: str) -> int | None:
        """The model property that a write changes: the root pose of a fixed base is the frame of the joint that fixes
        it, a joint property of the model."""
        if field == "root_link_pose_w" and not asset.root_view.is_floating_base:
            return self.joint_properties
        return None


def _newton_backend(sim: SimulationContext) -> _NewtonBackend | None:
    """Return Newton's fast path if the simulation runs on Newton, without importing Newton otherwise."""
    if type(sim.backend_manager).__module__.split(".")[0] != "isaaclab_newton":
        return None
    return _NewtonBackend()
