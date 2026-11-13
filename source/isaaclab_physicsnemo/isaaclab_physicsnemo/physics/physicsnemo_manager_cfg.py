# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the PhysicsNeMo learned-dynamics Newton solver."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

from isaaclab_newton.physics import NewtonSolverCfg

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .physicsnemo_manager import NewtonPhysicsNeMoManager


@configclass
class PhysicsNeMoSolverCfg(NewtonSolverCfg):
    """Run a saved PhysicsNeMo Robotics dynamics artifact as the Newton solver.

    Select it through :attr:`~isaaclab_newton.physics.NewtonCfg.solver_cfg`. The artifact replaces the
    complete Newton state transition, so the environment must use ``num_substeps = 1``, a physics step
    equal to the artifact frame duration, and ``use_cuda_graph = False``.
    :func:`~isaaclab_physicsnemo.apply_learned_physics` applies these settings to a recording
    configuration.
    """

    class_type: type[NewtonPhysicsNeMoManager] | str = "{DIR}.physicsnemo_manager:NewtonPhysicsNeMoManager"
    """Manager class for this solver."""

    solver_type: str = "physicsnemo"
    """Solver type metadata for logging."""

    artifact_path: str = MISSING
    """Directory of a saved :class:`~physicsnemo_robotics.dynamics.DynamicsArtifact`."""

    continuity: Literal["verify", "refill", "off"] = "verify"
    """Response to Newton state rewrites that were not reported as environment resets.

    ``"verify"`` rejects them, ``"refill"`` restarts the affected histories, and ``"off"`` trusts the
    reset masks Isaac Lab forwards.
    """

    inference_dtype: Literal["float32", "float16"] = "float32"
    """Precision of the network's projections. ``"float16"`` runs them under FP16 autocast, with attention scores
    and every state computation in FP32, which speeds up large models at a small cost in accuracy."""

    model_factory: Callable | str | None = None
    """Factory returning the network module, required only for artifacts storing a plain Torch module."""

    codec_factory: Callable | str | None = None
    """Factory rebuilding a custom state codec.

    Defaults to :func:`~isaaclab_physicsnemo.heading.artifact_codec_factory`, which rebuilds the codecs of
    this extension.
    """

    feature_factory: Callable | str | None = None
    """Factory rebuilding a custom feature encoder.

    Defaults to :func:`~isaaclab_physicsnemo.heading.artifact_feature_factory`, which rebuilds the encoders
    of this extension.
    """
