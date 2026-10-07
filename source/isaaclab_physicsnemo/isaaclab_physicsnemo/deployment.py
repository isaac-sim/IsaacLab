# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replace the teacher solver of an environment configuration with a dynamics artifact."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from isaaclab_newton.physics import NewtonCfg
from physicsnemo_robotics.dynamics import DynamicsArtifact

from isaaclab.utils import replace

from .physics import PhysicsNeMoSolverCfg

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnvCfg, ManagerBasedEnvCfg


def apply_learned_physics(
    env_cfg: DirectRLEnvCfg | ManagerBasedEnvCfg,
    artifact_path: str | Path,
    *,
    continuity: Literal["verify", "refill", "off"] = "verify",
) -> DynamicsArtifact:
    """Replace the teacher Newton solver of an environment configuration with a dynamics artifact.

    The configuration must be the one the artifact's training data was recorded with. Its Newton
    model settings, such as shape and collision defaults, are kept so that the deployed model has the
    trained identity. The physics step is set to the artifact frame duration, and the learned solver
    runs once per physics step with eager CUDA execution.

    Args:
        env_cfg: Environment configuration whose ``sim.physics`` is a :class:`NewtonCfg`. Modified in place.
        artifact_path: Directory of a saved :class:`~physicsnemo_robotics.dynamics.DynamicsArtifact`.
        continuity: Response of the learned solver to state rewrites that were not reported as resets.

    Returns:
        The loaded artifact, for use with :func:`~isaaclab_physicsnemo.preflight_learned_physics`.

    Raises:
        TypeError: If the configuration does not select the Newton physics backend.
    """
    if not isinstance(env_cfg.sim.physics, NewtonCfg):
        raise TypeError(f"Learned physics requires a NewtonCfg teacher configuration, got {type(env_cfg.sim.physics)}.")
    artifact_path = Path(artifact_path).expanduser().resolve()
    artifact = DynamicsArtifact.load(artifact_path)
    env_cfg.sim.dt = float(artifact.manifest.schema.simulation.time.frame_dt)
    # NewtonCfg derives ``class_type`` from ``solver_cfg``, so it must be cleared before rebuilding.
    env_cfg.sim.physics = replace(
        env_cfg.sim.physics,
        class_type=None,
        solver_cfg=PhysicsNeMoSolverCfg(artifact_path=str(artifact_path), continuity=continuity),
        num_substeps=1,
        use_cuda_graph=False,
        deterministic_mode="not_guaranteed",
    )
    return artifact
