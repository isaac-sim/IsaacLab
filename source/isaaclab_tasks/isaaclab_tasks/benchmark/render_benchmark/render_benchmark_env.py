# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct environment for renderer and physics-plus-renderer benchmarking."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import save_images_to_file

if TYPE_CHECKING:
    from .render_benchmark_env_cfg import RenderBenchmarkFrankaCabinetEnvCfg


class RenderBenchmarkEnv(DirectRLEnv):
    """Environment that animates its articulations so a renderer can be profiled on them.

    Every articulation in the configured scene is driven by a sinusoid around its default joint
    positions. Actions are ignored and rewards are zero: the only output that matters is the camera
    image, and the only cost that matters is the time spent producing it.

    How the sinusoid reaches the joints depends on
    :attr:`~.render_benchmark_env_cfg.RenderBenchmarkFrankaCabinetEnvCfg.benchmark_mode`. In
    ``"physics_render"`` mode the pose becomes an actuator position target before the physics
    steps, so the solver tracks it the way it would in an ordinary task. In ``"render"`` mode the
    pose is instead written straight into the simulation after the last physics step and before
    the camera is read, so the frame a renderer is timed on is the analytic pose rather than
    whatever the solver arrived at.
    """

    cfg: RenderBenchmarkFrankaCabinetEnvCfg

    def __init__(self, cfg: RenderBenchmarkFrankaCabinetEnvCfg, render_mode: str | None = None, **kwargs):
        # Per-(env, joint) sinusoid phases [rad] and the elapsed animation time [s]. The phases
        # are filled on the first step; see :meth:`_sample_animation_phases`.
        self._anim_phases: dict[str, torch.Tensor] | None = None
        self._anim_time: float = 0.0
        super().__init__(cfg, render_mode, **kwargs)
        self._tiled_camera = self.scene["tiled_camera"]

    # --- joint animation -----------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if self.cfg.joint_animation_amplitude <= 0.0:
            return
        if self._anim_phases is None:
            self._anim_phases = self._sample_animation_phases()
        self._anim_time += self.cfg.sim.dt * self.cfg.decimation
        # In render mode the pose is applied in _get_observations instead, once physics can no
        # longer move it. Only the actuated mode has anything to request before the solver runs.
        if self.cfg.benchmark_mode == "physics_render":
            self._request_joint_targets()

    def _apply_action(self) -> None:
        pass

    def _sample_animation_phases(self) -> dict[str, torch.Tensor]:
        """Derive a phase offset [rad] for every (env, joint) pair, keyed by articulation name.

        The phases come from a deterministic hash of the indices rather than the torch RNG so a
        given frame holds the same pose in every run. The default RNG state varies between
        processes, which would leave an OVRTX run and a Warp run rendering different poses at the
        same frame even though the solver itself is deterministic, making their timings
        incomparable. Multiplying by two coprime primes spreads the phases pseudo-uniformly.

        Deferred to the first step because it reads articulation data, which the simulation only
        populates after scene initialization.
        """
        phases = {}
        for name, articulation in self.scene.articulations.items():
            default_pos = articulation.data.default_joint_pos.torch
            num_envs, num_joints = default_pos.shape
            env_idx = torch.arange(num_envs, device=self.device, dtype=default_pos.dtype).unsqueeze(1)
            joint_idx = torch.arange(num_joints, device=self.device, dtype=default_pos.dtype).unsqueeze(0)
            phases[name] = ((env_idx * 7919.0 + joint_idx * 6553.0) % 10007.0) * (2.0 * math.pi / 10007.0)
        return phases

    def _animation_targets(self) -> Iterator[tuple[Articulation, torch.Tensor]]:
        """Yield every articulation with its joint pose for the current animation time.

        Yields:
            Each articulation and its per-joint target [m or rad, depending on joint type],
            clamped to the joint's soft limits, shape ``[num_envs, num_joints]``.
        """
        omega = 2.0 * math.pi * self.cfg.joint_animation_freq_hz
        for name, articulation in self.scene.articulations.items():
            default_pos = articulation.data.default_joint_pos.torch
            offset = self.cfg.joint_animation_amplitude * torch.sin(omega * self._anim_time + self._anim_phases[name])
            soft_limits = articulation.data.soft_joint_pos_limits.torch
            yield articulation, torch.clamp(default_pos + offset, soft_limits[..., 0], soft_limits[..., 1])

    def _request_joint_targets(self) -> None:
        """Ask the actuators to track the current pose, leaving the solver to reach it."""
        for articulation, target in self._animation_targets():
            articulation.actuators.target_command.set_position_index(value=target)

    def _pose_joints_directly(self) -> None:
        """Place the joints at the current pose without asking the solver to reach it.

        The velocity write keeps the articulation from carrying momentum across a pose it never
        integrated toward, which would otherwise show up as contact and joint-limit work in the
        step that follows.
        """
        for articulation, target in self._animation_targets():
            articulation.write_joint_position_to_sim_index(position=target)
            articulation.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(target))

    # --- DirectRLEnv plumbing ------------------------------------------------

    def _get_observations(self) -> dict:
        # In render mode the pose is applied here rather than in _pre_physics_step. Physics runs
        # between those two points, and its drives, gravity and joint limits would all pull the
        # joints off a pose written beforehand -- the renderer would then be timed on the solver's
        # output rather than on the analytic one. Writing it here, after the last physics step and
        # before the render below, is what makes the rendered frame the pose this mode advertises.
        # forward() propagates the joint write to the body transforms the renderer reads without
        # stepping the solver again.
        if self.cfg.benchmark_mode == "render" and self._anim_phases is not None:
            self._pose_joints_directly()
            self.sim.forward()
            # The renderer syncs scene state at most once per physics step, and forward() does not
            # advance that count. With lazy_sensor_update off, InteractiveScene.update has already
            # synced this step, so without clearing the dedupe the render below would reuse the
            # transforms captured before the write above -- the solver's pose, not this one.
            self.sim.render_context.reset_scene_state_cadence()

        # Sensor buffers update lazily, so reading the camera's data is what drives the render.
        # This access is the work the benchmark measures: keep it unconditional even when no
        # image is written, or the profile records a scene that was never rendered.
        output = self._tiled_camera.data.output
        if self.cfg.write_image_to_file:
            self._write_camera_image(output)
        return {"policy": torch.zeros((self.num_envs, 1), device=self.device)}

    def _write_camera_image(self, output: dict) -> None:
        """Dump the current camera frame to a PNG.

        Purely a debugging aid, so any failure is reported and swallowed rather than allowed to
        abort a benchmark run.

        Args:
            output: Rendered outputs from the tiled camera, keyed by data type.
        """
        data_type = self.cfg.scene.tiled_camera.data_types[0]
        if not isinstance(output, dict) or data_type not in output:
            return
        try:
            image = output[data_type]
            if not torch.is_tensor(image):
                image = torch.from_dlpack(image)
            image = image.float()
            # Renderers hand back either 0-255 or already-normalized data.
            if image.max() > 1.5:
                image = image / 255.0
            save_images_to_file(image[:, ..., :3], f"render_benchmark_{data_type}.{self.common_step_counter:06d}.png")
        except Exception as error:
            print(f"[render_benchmark] write_image_to_file failed: {error}")

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(time_out), time_out
