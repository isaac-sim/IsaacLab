# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record Isaac Lab Newton episodes as PhysicsNeMo trajectory datasets."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import torch
from physicsnemo_robotics.data import NewtonTrajectoryWriter, TrajectoryRecorder
from physicsnemo_robotics.dynamics import StateCodec, TrajectoryFieldSpec, TrajectorySchema, newton_state_codec
from physicsnemo_robotics.newton import NewtonStepContext, NewtonTimeSpec, field_to_torch

from .binding import NewtonBinding
from .physics import validate_effort_control

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import DirectRLEnv, ManagerBasedEnv

CONTROL_KEY = "control/joint_f"
"""Trajectory field holding the joint efforts that Isaac Lab applied during each transition."""

JOINT_TARGET_KEY = "control/joint_target_q"
"""Trajectory field holding the Newton joint-position targets commanded during each control-period transition."""


class RecordedFeature(Protocol):
    """Transition input recorded next to the state, such as a PhysicsNeMo Robotics fixed-contact encoder."""

    @property
    def key(self) -> str:
        """Trajectory field name."""
        ...

    @property
    def field_spec(self) -> TrajectoryFieldSpec:
        """Declaration of the transition-aligned trajectory field."""
        ...

    def encode_step(self, context: NewtonStepContext) -> torch.Tensor:
        """Encode the field for the transition that starts at ``context.state_in``."""
        ...


class TrajectoryDatasetRecorder:
    """Write the episodes of a running Isaac Lab environment to a Zarr trajectory dataset.

    The recorder reads the generalized joint state of every environment after each
    :meth:`~isaaclab.envs.DirectRLEnv.step` and stores the applied joint efforts
    (:data:`CONTROL_KEY`) as the transition between consecutive states. One environment step must
    equal one physics step, so the environment must use ``decimation = 1``. Isaac Lab resets finished
    environments before ``step`` returns, so each finished episode drops its last observed frame and a
    new episode starts from the returned reset state.

    With ``control_targets``, a frame is instead one environment step of any decimation, and the
    transition stores the joint-position targets of that articulation in the layout of Newton's
    ``control.joint_target_q`` (:data:`JOINT_TARGET_KEY`). The model then learns the actuator response
    within the control period, as FlashNeRD does, and deploys in a scene whose actuators only forward
    their targets (:class:`~isaaclab.actuators.ImplicitActuatorCfg` with zero gains).

    The applied joint efforts must be the only actuation and external input of the scene: implicit
    position drives are rejected, and the environment must not apply external body wrenches, whose
    effect would be recorded without its cause.

    Use it as a context manager so that pending episodes are written and the dataset is published.
    """

    def __init__(
        self,
        env: DirectRLEnv | ManagerBasedEnv,
        path: str | Path,
        *,
        scenario_id: str,
        remove_root_translation: bool | Literal["planar"] = False,
        codec: StateCodec | None = None,
        recorded_features: Sequence[RecordedFeature] = (),
        control_targets: Articulation | None = None,
        seed: int = 0,
        replace: bool = False,
    ) -> None:
        """Bind the initialized environment and create the dataset.

        Args:
            env: Environment after its first reset, running a Newton teacher solver.
            path: Output dataset directory.
            scenario_id: Stable name of the data-collection scenario stored with every episode.
            remove_root_translation: Root translation removed from model inputs. Use ``"planar"`` for
                floating-base robots so that per-environment origins do not reach the model. Ignored when
                ``codec`` is given.
            codec: Joint state codec of the model inputs, such as
                :class:`~isaaclab_physicsnemo.HeadingJointStateCodec`. Defaults to the PhysicsNeMo Robotics
                joint codec.
            recorded_features: Encoders of additional transition inputs, such as PhysicsNeMo Robotics
                fixed contact probes. Each transition stores their values for its starting state.
            control_targets: Articulation whose joint-position targets are the control of each
                environment step. Defaults to recording the applied joint efforts of each physics step.
            seed: Seed stored with every episode.
            replace: Whether to replace an existing dataset at ``path``.

        Raises:
            ValueError: If one environment step spans more than one physics step without
                ``control_targets``, or a joint uses an implicit position drive.
        """
        manager = env.sim.physics_manager
        decimation = round(env.step_dt / manager.get_dt())
        if not math.isclose(env.step_dt, decimation * manager.get_dt(), rel_tol=1e-9) or (
            decimation != 1 and control_targets is None
        ):
            raise ValueError(
                f"Recording efforts requires one physics step per environment step, got step_dt={env.step_dt} and"
                f" physics dt={manager.get_dt()}. Set decimation to 1, or record control targets."
            )
        self.binding = NewtonBinding(manager)
        validate_effort_control(self.binding.model)
        spec = self.binding.spec
        if control_targets is not None:
            time = NewtonTimeSpec(solver_dt=spec.time.solver_dt, substeps=spec.time.substeps * decimation)
            spec = dataclasses.replace(spec, time=time)
            self._control_key, self._control = JOINT_TARGET_KEY, _JointTargets(control_targets, self.binding)
        else:
            self._control_key, self._control = CONTROL_KEY, self._applied_joint_f
        self.codec = (
            codec
            if codec is not None
            else newton_state_codec(self.binding.model, "joint", joint_remove_root_translation=remove_root_translation)
        )
        self.schema = TrajectorySchema.from_specs(
            spec,
            self.binding.execution_spec,
            self.codec,
            fields=(
                TrajectoryFieldSpec(
                    self._control_key, "transition", dtype="float32", trailing_shape=(self._control().shape[-1],)
                ),
                *(feature.field_spec for feature in recorded_features),
            ),
        )
        self._recorded_features = tuple(recorded_features)
        self._frame_dt = env.step_dt
        self._recorder = TrajectoryRecorder(
            self.schema,
            worlds=self.binding.worlds,
            seed=seed,
            scenario_id=scenario_id,
            metadata={"decimation": decimation, "physics_dt": manager.get_dt()},
        )
        self._writer = NewtonTrajectoryWriter(path, self.schema, mode="replace" if replace else "create")
        self._recorder.begin(self.codec.read(self.binding.state))
        self._pending_features = self._encode_features()
        self.episodes_written = 0
        """Number of episodes written to the dataset."""

    def __enter__(self) -> TrajectoryDatasetRecorder:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.close()
        else:
            # Discard a partially written replacement dataset instead of publishing it.
            self._writer.__exit__(exc_type, exc, traceback)

    def record(self, dones: torch.Tensor) -> None:
        """Record the transition produced by the latest environment step.

        Args:
            dones: Terminated or truncated environments returned by the step, shape [num_envs].
        """
        transitions = {self._control_key: self._control(), **self._pending_features}
        self._recorder.record(self.codec.read(self.binding.state), transitions=transitions)
        if bool(dones.any()):
            self._write(self._recorder.finalize(dones, drop_last=1))
            self._recorder.begin(self.codec.read(self.binding.state), worlds_mask=dones)
        # The current state, reset or not, starts the next transition.
        self._pending_features = self._encode_features()

    def close(self) -> Path:
        """Write the open episodes and publish the dataset.

        Returns:
            The dataset directory.
        """
        self._write(self._recorder.finalize())
        return self._writer.close()

    def _encode_features(self) -> dict[str, torch.Tensor]:
        """Encode the recorded features of the current state."""
        if not self._recorded_features:
            return {}
        state = self.binding.state
        context = NewtonStepContext(
            state_in=state,
            state_out=state,
            control=self.binding.control,
            contacts=self.binding.contacts,
            frame_index=0,
            substep_index=0,
            solver_dt=self._frame_dt,
            frame_dt=self._frame_dt,
        )
        return {feature.key: feature.encode_step(context).clone() for feature in self._recorded_features}

    def _applied_joint_f(self) -> torch.Tensor:
        return field_to_torch(self.binding.control.joint_f, clone=True).reshape(self.binding.worlds, -1)

    def _write(self, episodes) -> None:
        for episode in episodes:
            self._writer.append(episode.data, episode.metadata)
            self.episodes_written += 1


class _JointTargets:
    """Joint-position targets of an articulation in the layout of Newton's ``control.joint_target_q``.

    Coordinates that the articulation does not command keep the control's own values, such as the root
    targets, so recorded vectors equal what a target-forwarding deployment writes.
    """

    def __init__(self, articulation: Articulation, binding: NewtonBinding) -> None:
        model = binding.model
        if binding.control.joint_target_q is None:
            raise ValueError("The Newton control has no joint-position targets.")
        world, starts = model.joint_world.numpy(), model.joint_q_start.numpy()
        coordinates = {
            str(model.joint_label[joint]).rsplit("/", 1)[-1]: int(starts[joint])
            for joint in range(model.joint_count)
            if world[joint] == 0
        }
        missing = [name for name in articulation.joint_names if name not in coordinates]
        if missing:
            raise ValueError(f"The Newton model has no joints named {missing}.")
        self._articulation, self._binding = articulation, binding
        self._columns = torch.tensor([coordinates[name] for name in articulation.joint_names], device=binding.device)

    def __call__(self) -> torch.Tensor:
        targets = field_to_torch(self._binding.control.joint_target_q, clone=True).reshape(self._binding.worlds, -1)
        targets[:, self._columns] = self._articulation.data.joint_pos_target.torch.to(targets.dtype)
        return targets
