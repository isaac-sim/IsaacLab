# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lifecycle bootstrap for an application-owned OVRTX OVXR pair."""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING, Any

from isaaclab_ov.renderers import OvLibrariesRenderLease, OVRTXRenderer, OVRTXRendererCfg

from isaaclab.sim import SimulationContext

_TERMINAL_MODE_SENTINEL = "ISAAC_LAUNCHED_FROM_TERMINAL"

if TYPE_CHECKING:
    import numpy as np
    from isaaclab_ov.renderers.render_lease import OvLibrariesScenePublisher

    from isaaclab.sim import SimulationCfg


def validate_ovxr_session_preparation() -> None:
    """Reject ineligible preparation before the caller changes renderer globals."""
    if SimulationContext.instance() is not None:
        raise RuntimeError("OVXR bootstrap must create the SimulationContext immediately before the environment")
    if hasattr(builtins, _TERMINAL_MODE_SENTINEL) and getattr(builtins, _TERMINAL_MODE_SENTINEL) is not True:
        raise RuntimeError("OVXR bootstrap cannot reuse a context in extension mode")


class OvxrSessionBootstrap:
    """Reserve an OVRTX/OVStage lease before a real environment is constructed."""

    def __init__(self, sim_cfg: SimulationCfg, renderer_cfg: OVRTXRendererCfg | None = None) -> None:
        validate_ovxr_session_preparation()

        self._sim: SimulationContext | None = None
        self._lease: OvLibrariesRenderLease | None = None
        self._owner_transfer_capability: object | None = None
        self._bound = False
        self._closed = False
        self._installed_terminal_mode_sentinel = False
        self._target_frame_readers: dict[str, tuple[Any, Any, Any]] = {}

        if not hasattr(builtins, _TERMINAL_MODE_SENTINEL):
            setattr(builtins, _TERMINAL_MODE_SENTINEL, True)
            self._installed_terminal_mode_sentinel = True

        try:
            sim = SimulationContext(sim_cfg)
            # The simulation registry owns the renderer, including cloning and
            # cleanup; the XR session only borrows its exact native pair.
            renderer = sim.get_or_create_backend(renderer_cfg or OVRTXRendererCfg())
            if not isinstance(renderer, OVRTXRenderer):
                raise TypeError(f"OVRTXRendererCfg resolved to unexpected renderer {type(renderer).__name__}")
            lease = renderer.acquire_ov_libraries_render_lease()
        except BaseException as preparation_error:
            sim = SimulationContext.instance()
            try:
                if sim is not None:
                    sim.clear_instance()
            except BaseException as cleanup_error:
                preparation_error.add_note(f"Cleanup after failed XR construction also failed: {cleanup_error!r}")
            finally:
                self._restore_terminal_mode_sentinel()
            raise

        self._sim = sim
        self._lease = lease

    def capture_scene_publisher(self) -> OvLibrariesScenePublisher:
        """Capture the application scene publisher before render ownership transfers."""
        self._require_open()
        if self._bound or self._lease is None:
            raise RuntimeError("Scene publisher is available only before OVXR Stage binding")
        return self._lease.capture_scene_publisher()

    @property
    def simulation_context(self) -> SimulationContext:
        """Return the exact pre-created simulation context."""
        self._require_open()
        assert self._sim is not None
        return self._sim

    def bind_environment(self, env: Any) -> OvLibrariesRenderLease:
        """Prepare and bind the completed environment Stage to the reserved pair."""
        self._require_open()
        if self._bound:
            raise RuntimeError("OVXR bootstrap is already bound")
        try:
            unwrapped = getattr(env, "unwrapped", env)
            if unwrapped.sim is not self._sim:
                raise RuntimeError("Environment did not reuse the exact pre-created SimulationContext")
            scene = unwrapped.scene
            if scene.stage is None:
                raise RuntimeError("Environment has no completed InteractiveScene Stage")
            num_envs = unwrapped.num_envs
            if not isinstance(num_envs, int) or num_envs <= 0:
                raise RuntimeError(f"Environment has invalid num_envs: {num_envs!r}")
            assert self._sim is not None and self._lease is not None
            self._sim.render_context.ensure_prepare_stage(scene.stage, num_envs)
            self._lease.validate_live()
            self._bound = True
            return self._lease
        finally:
            self._restore_terminal_mode_sentinel()

    def offer_render_owner_transfer(self) -> object:
        """Offer the exact unbound OVXR lease before Stage binding."""
        self._require_open()
        if self._bound or self._lease is None:
            raise RuntimeError("Owner transfer is available only before OVXR Stage binding")
        capability = self._lease.offer_owner_transfer()
        self._owner_transfer_capability = capability
        return capability

    def claim_render_owner_transfer(self, capability: object) -> OvLibrariesRenderLease:
        """Claim the offered lease on the worker that will bind, use, and close it."""
        self._require_open()
        assert self._lease is not None
        self._lease.claim_owner_transfer(capability)
        self._owner_transfer_capability = None
        return self._lease

    def get_world_transform(self, prim_path: str) -> np.ndarray:
        """Read target-to-world transform [4, 4], float64, with translation in scene units."""
        self._require_open()
        if not self._bound or self._sim is None:
            raise RuntimeError("Target-frame transforms require a bound OVXR environment")
        if not isinstance(prim_path, str) or not prim_path.startswith("/"):
            raise ValueError(f"Target-frame prim path must be absolute: {prim_path!r}")

        import numpy as np
        import warp as wp

        from isaaclab.scene_data import SceneDataFormat

        provider = self._sim.get_scene_data_provider()
        cached = self._target_frame_readers.get(prim_path)
        if cached is None or cached[0] is not provider:
            transform_paths = list(provider.backend.transform_paths)
            match_count = transform_paths.count(prim_path)
            if match_count != 1:
                raise RuntimeError(
                    f"Target-frame prim path {prim_path!r} matched {match_count} live backend transforms"
                )
            mapping = provider.create_mapping([prim_path])
            output = SceneDataFormat.Matrix44()
            cached = (provider, mapping, output)
            self._target_frame_readers[prim_path] = cached

        provider, mapping, output = cached
        if not provider.get_transforms(output, mapping=mapping, allow_passthrough=False, count=1):
            raise RuntimeError(f"No SceneData transform conversion is available for target frame {prim_path!r}")
        if output.matrices is None:
            raise RuntimeError(f"SceneData returned no matrix buffer for target frame {prim_path!r}")

        wp.synchronize_device(str(output.matrices.device))
        world_T_target = np.asarray(output.matrices.numpy()[0], dtype=np.float64)
        if world_T_target.shape != (4, 4) or not np.all(np.isfinite(world_T_target)):
            raise RuntimeError(f"Target-frame transform {prim_path!r} is missing or non-finite")
        if not np.allclose(world_T_target[3], (0.0, 0.0, 0.0, 1.0), atol=1e-5, rtol=0.0):
            raise RuntimeError(f"Target-frame transform {prim_path!r} is not homogeneous")
        return world_T_target.copy()

    def get_target_frame_T_world(self, prim_path: str) -> np.ndarray:
        """Read world-to-target transform [4, 4], float32, with translation in scene units."""
        import numpy as np

        world_T_target = self.get_world_transform(prim_path)
        try:
            target_T_world = np.linalg.inv(world_T_target)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(f"Target-frame transform {prim_path!r} is singular") from exc
        if not np.all(np.isfinite(target_T_world)):
            raise RuntimeError(f"Inverse target-frame transform {prim_path!r} is non-finite")
        return target_T_world.astype(np.float32)

    @property
    def closed(self) -> bool:
        """Whether the lease and its retained publication resources retired successfully."""
        return self._closed

    def close(self) -> None:
        """Release after every borrower is destroyed; retain ownership for retry on failure."""
        if self._closed:
            return
        # Keep ownership and the bootstrap retryable until retirement succeeds.
        if self._lease is not None:
            if self._owner_transfer_capability is not None:
                self._lease.cancel_owner_transfer(self._owner_transfer_capability)
                self._owner_transfer_capability = None
            self._lease.release()
        self._target_frame_readers.clear()
        self._restore_terminal_mode_sentinel()
        self._closed = True

    def abort(self) -> None:
        """Release a failed construction; retain its context if lease retirement fails."""
        if self._closed:
            return
        self.close()
        if not self._bound and self._sim is not None and SimulationContext.instance() is self._sim:
            self._sim.clear_instance()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("OVXR bootstrap is closed")

    def _restore_terminal_mode_sentinel(self) -> None:
        if self._installed_terminal_mode_sentinel:
            if hasattr(builtins, _TERMINAL_MODE_SENTINEL):
                delattr(builtins, _TERMINAL_MODE_SENTINEL)
            self._installed_terminal_mode_sentinel = False
