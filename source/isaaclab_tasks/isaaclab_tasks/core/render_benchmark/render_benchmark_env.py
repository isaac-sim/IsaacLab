# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct environment for render-only benchmarking."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import Camera, save_images_to_file

if TYPE_CHECKING:
    from .render_benchmark_env_cfg import RenderBenchmarkFrankaCabinetEnvCfg


class RenderBenchmarkEnv(DirectRLEnv):
    """Environment that animates its articulations so a renderer can be profiled on them.

    Every articulation in :attr:`RenderBenchmarkFrankaCabinetEnvCfg.articulations` is driven by a
    sinusoid around its default joint positions. Actions are ignored and rewards are zero: the
    only output that matters is the camera image, and the only cost that matters is the time
    spent producing it.
    """

    cfg: RenderBenchmarkFrankaCabinetEnvCfg

    # --- scene construction --------------------------------------------------

    def _setup_scene(self):
        # Per-(env, joint) sinusoid phases [rad] and the elapsed animation time [s]. The phases
        # are filled on the first step; see :meth:`_sample_animation_phases`.
        self._anim_phases: dict[str, torch.Tensor] | None = None
        self._anim_time: float = 0.0

        self._articulations: dict[str, Articulation] = {
            name: Articulation(articulation_cfg) for name, articulation_cfg in self.cfg.articulations.items()
        }
        self._tiled_camera = Camera(self.cfg.tiled_camera)
        self._ground = RigidObject(self._ground_cfg())
        self._spawn_lights()

        # Assets must exist before planning: the plan keys its rows off the cfgs they queued.
        src, dest = "/World/envs/env_0", "/World/envs/env_{}"
        pos = cloner.grid_transforms(self.scene.num_envs, self.scene.cfg.env_spacing)[0]
        plan = cloner.clone_plan_from_env_0(src, dest, self.scene.num_envs, pos)
        cloner.replicate(plan)

        # PhysX replication requires explicit collision filtering between environments.
        if "physx" in self.scene.physics_backend:
            self.scene.filter_collisions(global_prim_paths=[])

        for name, articulation in self._articulations.items():
            self.scene.articulations[name] = articulation
        self.scene.rigid_objects["ground"] = self._ground
        self.scene.sensors["tiled_camera"] = self._tiled_camera

    def _ground_cfg(self) -> RigidObjectCfg:
        """Build the per-environment ground cuboid.

        A flat cuboid routed through :class:`~isaaclab.assets.RigidObject` keeps the comparison
        apples-to-apples: a USD ground plane is invisible to Warp's renderer, which only knows
        about simulation meshes, so the two renderers would otherwise disagree on primary-ray
        misses and shadow-ray counts.
        """
        size_x, size_y = self.cfg.ground_size
        thickness = self.cfg.ground_thickness
        return RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Ground",
            spawn=sim_utils.CuboidCfg(
                size=(size_x, size_y, thickness),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=self.cfg.ground_color, metallic=0.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, self.cfg.ground_top_z - thickness / 2.0)),
        )

    def _spawn_lights(self) -> None:
        """Spawn the ambient dome light and, when configured, the directional light above it."""
        dome_cfg = sim_utils.DomeLightCfg(intensity=self.cfg.dome_light_intensity, color=(0.75, 0.75, 0.75))
        dome_cfg.func("/World/Light", dome_cfg)
        if self.cfg.light_cfg is not None:
            self.cfg.light_cfg.func(
                "/World/LightDirectional",
                self.cfg.light_cfg,
                orientation=self.cfg.light_orientation,
            )

    # --- joint animation -----------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if self.cfg.joint_animation_amplitude <= 0.0:
            return
        if self._anim_phases is None:
            self._anim_phases = self._sample_animation_phases()
        self._animate_joints()

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
        populates after :meth:`_setup_scene` returns.
        """
        phases = {}
        for name, articulation in self._articulations.items():
            default_pos = articulation.data.default_joint_pos.torch
            num_envs, num_joints = default_pos.shape
            env_idx = torch.arange(num_envs, device=self.device, dtype=default_pos.dtype).unsqueeze(1)
            joint_idx = torch.arange(num_joints, device=self.device, dtype=default_pos.dtype).unsqueeze(0)
            phases[name] = ((env_idx * 7919.0 + joint_idx * 6553.0) % 10007.0) * (2.0 * math.pi / 10007.0)
        return phases

    def _animate_joints(self) -> None:
        """Advance the animation clock and drive every joint to its sinusoidal target."""
        self._anim_time += self.cfg.sim.dt * self.cfg.decimation
        omega = 2.0 * math.pi * self.cfg.joint_animation_freq_hz
        for name, articulation in self._articulations.items():
            default_pos = articulation.data.default_joint_pos.torch
            offset = self.cfg.joint_animation_amplitude * torch.sin(omega * self._anim_time + self._anim_phases[name])
            soft_limits = articulation.data.soft_joint_pos_limits.torch
            target = torch.clamp(default_pos + offset, soft_limits[..., 0], soft_limits[..., 1])
            articulation.actuators.target_command.set_position_index(value=target)

    # --- DirectRLEnv plumbing ------------------------------------------------

    def _get_observations(self) -> dict:
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
        data_type = self.cfg.tiled_camera.data_types[0]
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
