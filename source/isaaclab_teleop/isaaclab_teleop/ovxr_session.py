# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lifecycle bootstrap for an application-owned OVRTX OVXR pair."""

from __future__ import annotations

import builtins
import contextlib
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from isaaclab.app import get_settings_manager
from isaaclab.scene_data import SceneDataFormat
from isaaclab.sim import SimulationContext

from .ovxr_renderer import OvxrRenderer, OvxrRendererCfg, OvxrRenderLease

logger = logging.getLogger(__name__)

_TERMINAL_MODE_SENTINEL = "ISAAC_LAUNCHED_FROM_TERMINAL"

_RETAINED_NATIVE_PAIRS: list[tuple[Any, ...]] = []
"""Native renderer and Stage resources whose retirement was never proved; they outlive every session."""

if TYPE_CHECKING:
    from isaaclab.sensors import CameraCfg
    from isaaclab.sim import SimulationCfg

    from .ovxr_renderer import OvxrScenePublisher


def validate_ovxr_session_preparation() -> None:
    """Reject ineligible preparation before the caller changes renderer globals."""
    if SimulationContext.instance() is not None:
        raise RuntimeError("OVXR bootstrap must create the SimulationContext immediately before the environment")
    if hasattr(builtins, _TERMINAL_MODE_SENTINEL) and getattr(builtins, _TERMINAL_MODE_SENTINEL) is not True:
        raise RuntimeError("OVXR bootstrap cannot reuse a context in extension mode")


def _resolve_cuda_device(device: str) -> str:
    """Resolve a bare ``"cuda"`` device to the GPU that the simulation selects for it.

    :class:`~isaaclab.sim.SimulationContext` and :func:`~isaaclab.app.launch_simulation` read
    the ``/physics/cudaDevice`` setting the same way.

    Args:
        device: Simulation device, for example ``"cpu"``, ``"cuda"`` or ``"cuda:1"``.

    Returns:
        ``"cuda:<index>"`` for a bare ``"cuda"``; any other device unchanged.
    """
    if device != "cuda":
        return device
    cuda_device = get_settings_manager().get("/physics/cudaDevice")
    return f"cuda:{max(0, int(cuda_device) if cuda_device is not None else 0)}"


class OvxrSessionBootstrap:
    """Reserve an OVRTX/OVStage lease before a real environment is constructed.

    The bootstrap creates the :class:`~isaaclab.sim.SimulationContext` from the ``sim_cfg``
    given here, and the environment reuses it. Only environments that accept a pre-created
    context are supported (manager-based environments; :class:`~isaaclab.envs.DirectRLEnv`
    always creates its own). :meth:`validate_environment` requires ``num_envs == 1`` and
    rejects a simulation device or time step that disagrees with the prepared context.
    """

    def __init__(self, sim_cfg: SimulationCfg, renderer_cfg: OvxrRendererCfg | None = None) -> None:
        validate_ovxr_session_preparation()
        if renderer_cfg is not None and not isinstance(renderer_cfg, OvxrRendererCfg):
            raise TypeError(
                "OVXR renders with OvxrRendererCfg; give the session and its camera panels the same configuration"
            )

        self._sim: SimulationContext | None = None
        self._renderer: OvxrRenderer | None = None
        self._lease: OvxrRenderLease | None = None
        self._retained_pair: tuple[Any, ...] = ()
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
            self._renderer_cfg = renderer_cfg or OvxrRendererCfg()
            renderer = sim.get_or_create_backend(self._renderer_cfg)
            if not isinstance(renderer, OvxrRenderer):
                raise TypeError(f"OvxrRendererCfg resolved to unexpected renderer {type(renderer).__name__}")
            # Camera sensors prepare every registered renderer on the application thread during
            # environment construction. The leased pair only records that call; the claimed render
            # owner completes it in bind_environment().
            lease = renderer.acquire_render_lease()
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
        self._prepared_device = sim.cfg.device
        self._prepared_dt = sim.cfg.dt
        self._renderer = renderer
        self._lease = lease

    @property
    def simulation_context(self) -> SimulationContext:
        """Return the exact pre-created simulation context."""
        self._require_open()
        assert self._sim is not None
        return self._sim

    def create_native_pair(self) -> None:
        """Create the session's OVStage and OVRTX engine, after renderer settings and before the environment."""
        self._require_open()
        assert self._lease is not None
        self._lease.create_native_pair()

    @property
    def closed(self) -> bool:
        """Whether the lease and its retained publication resources retired successfully."""
        return self._closed

    def capture_scene_publisher(self) -> OvxrScenePublisher:
        """Capture the application scene publisher before render ownership transfers."""
        self._require_open()
        if self._bound or self._lease is None:
            raise RuntimeError("Scene publisher is available only before OVXR Stage binding")
        return self._lease.capture_scene_publisher()

    def configure_auxiliary_camera(self, camera_cfg: CameraCfg, *, max_fps: float | None = None) -> None:
        """Authorize a camera with an update cap [Hz]; ``None`` renders it with every frame of its product."""
        self._require_open()
        if self._bound or self._lease is None:
            raise RuntimeError("Auxiliary camera must be configured before OVXR Stage binding")
        # The simulation shares a renderer only between equal configurations of the same type.
        panel_cfg = camera_cfg.renderer_cfg
        if type(panel_cfg) is not type(self._renderer_cfg) or panel_cfg != self._renderer_cfg:
            raise ValueError("A camera panel must use the OVXR session's renderer configuration")
        self._lease.configure_auxiliary_camera(camera_cfg, max_fps=max_fps)

    def validate_environment(self, env: Any) -> None:
        """Check that a constructed environment uses the prepared simulation context.

        Args:
            env: Constructed environment, possibly wrapped.

        Raises:
            RuntimeError: If the environment did not reuse the prepared context, has no
                completed stage or not exactly one environment, or its simulation
                configuration selects a different device or physics time step [s].
        """
        self._require_open()
        unwrapped = getattr(env, "unwrapped", env)
        if unwrapped.sim is not self._sim:
            raise RuntimeError("Environment did not reuse the exact pre-created SimulationContext")
        if unwrapped.scene.stage is None:
            raise RuntimeError("Environment has no completed InteractiveScene Stage")
        if unwrapped.num_envs != 1:
            raise RuntimeError(
                f"The OVXR render lease requires exactly one environment; got num_envs={unwrapped.num_envs!r}"
            )
        env_sim_cfg = unwrapped.cfg.sim
        for field, env_value, prepared_value in (
            ("device", _resolve_cuda_device(env_sim_cfg.device), self._prepared_device),
            ("dt", env_sim_cfg.dt, self._prepared_dt),
        ):
            if env_value != prepared_value:
                raise RuntimeError(
                    f"Environment sim.{field}={getattr(env_sim_cfg, field)!r} differs from the OVXR"
                    f" prepared simulation context ({prepared_value!r}); prepare the"
                    " session with the environment's sim configuration"
                )

    def bind_environment(self, env: Any) -> OvxrRenderLease:
        """Prepare and bind the completed environment Stage to the reserved pair.

        The bootstrap defers Stage materialization to the render owner, so call this on the
        worker after :meth:`claim_render_owner_transfer`.

        Args:
            env: Constructed environment, possibly wrapped.

        Returns:
            The bound lease.

        Raises:
            RuntimeError: If the bootstrap is closed or already bound, the owner transfer has not
                been claimed, or :meth:`validate_environment` rejects the environment.
        """
        self._require_open()
        if self._bound:
            raise RuntimeError("OVXR bootstrap is already bound")
        try:
            self.validate_environment(env)
            unwrapped = getattr(env, "unwrapped", env)
            scene = unwrapped.scene
            num_envs = unwrapped.num_envs
            assert self._sim is not None and self._lease is not None
            if not self._lease.materialize_deferred_stage(scene.stage, num_envs):
                self._sim.render_context.ensure_prepare_stage(scene.stage, num_envs)
            self._lease.validate_live()
            self._bound = True
            return self._lease
        finally:
            self._restore_terminal_mode_sentinel()

    def configure_view_partitions(self, camera_paths: tuple[str, ...], ledger: Any) -> None:
        """Keep tracking markers in the eye views and task geometry/lights visible to every camera.

        Call once on the render owner after binding, under the lease's owner lane. Only the
        session's presentation Stage changes; the application USD and ordinary renderers do not.
        """
        self._require_open()
        if not self._bound or self._sim is None or self._lease is None:
            raise RuntimeError("View partitions require a bound OVXR environment")
        self._lease.validate_live()
        import ovstage

        from .ovxr_scene import _resolve_env_light_partitions

        renderer = self._sim.get_or_create_backend(self._renderer_cfg)
        scene = renderer.scene
        usd_stage = self._sim.stage
        light_paths = list(_resolve_env_light_partitions(usd_stage, scene.clone_env_paths, copies=scene.clone_copies))
        settings_path = str(usd_stage.GetMetadata("renderSettingsPrimPath") or "/Render/IsaacLabSettings")
        # OVStage TOKEN_ID uses zero for the empty token (common, unpartitioned geometry).
        writes = [
            (
                scene.clone_env_paths,
                "primvars:omni:scenePartition",
                np.zeros(len(scene.clone_env_paths), dtype=np.uint64),
            ),
            (light_paths, "omni:scenePartition", np.zeros(len(light_paths), dtype=np.uint64)),
            (
                camera_paths,
                "omni:scenePartition",
                np.full(len(camera_paths), scene.paths.intern_token("viewer").raw_id, dtype=np.uint64),
            ),
            (
                [settings_path],
                "omni:rtx:scenePartitioning:showAllPartitionsByDefault",
                np.array([False], dtype=np.bool_),
            ),
        ]
        ordinal = ledger.reserve()
        try:
            for prim_paths, attribute, values in writes:
                if not prim_paths:
                    continue
                with scene.query(prim_paths) as query:
                    scene.stage.write_attribute(
                        query,
                        attribute,
                        ordinal=ordinal,
                        tensors=values,
                        is_array=False,
                        semantic=(
                            ovstage.AttributeSemantic.NONE
                            if values.dtype == np.bool_
                            else ovstage.AttributeSemantic.TOKEN_ID
                        ),
                    ).wait()
            ledger.publish(ordinal)
        except BaseException as error:
            # Startup interruption also must resolve the reservation before Stage retirement.
            try:
                ledger.abandon(ordinal)
            except BaseException as abandon_error:
                error.add_note(f"View-partition initialization cleanup also failed: {abandon_error!r}")
            raise

    def offer_render_owner_transfer(self) -> object:
        """Offer the exact unbound OVXR lease before Stage binding."""
        self._require_open()
        if self._bound or self._lease is None:
            raise RuntimeError("Owner transfer is available only before OVXR Stage binding")
        capability = self._lease.offer_owner_transfer()
        self._owner_transfer_capability = capability
        return capability

    def claim_render_owner_transfer(self, capability: object) -> OvxrRenderLease:
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

        # SceneData converts on the device's current stream. Wait only for that stream so a
        # publication does not serialize with unrelated work (such as in-flight XR rendering).
        if output.matrices.device.is_cuda:
            wp.synchronize_stream(wp.get_stream(output.matrices.device))
        world_T_target = np.asarray(output.matrices.numpy()[0], dtype=np.float64)
        if world_T_target.shape != (4, 4) or not np.all(np.isfinite(world_T_target)):
            raise RuntimeError(f"Target-frame transform {prim_path!r} is missing or non-finite")
        if not np.allclose(world_T_target[3], (0.0, 0.0, 0.0, 1.0), atol=1e-5, rtol=0.0):
            raise RuntimeError(f"Target-frame transform {prim_path!r} is not homogeneous")
        return world_T_target.copy()

    def get_target_frame_T_world(self, prim_path: str) -> np.ndarray:
        """Read world-to-target transform [4, 4], float32, with translation in scene units."""
        world_T_target = self.get_world_transform(prim_path)
        try:
            target_T_world = np.linalg.inv(world_T_target)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(f"Target-frame transform {prim_path!r} is singular") from exc
        if not np.all(np.isfinite(target_T_world)):
            raise RuntimeError(f"Inverse target-frame transform {prim_path!r} is non-finite")
        return target_T_world.astype(np.float32)

    def close(self) -> None:
        """Release after the OVXR context is destroyed; retain ownership for retry on failure."""
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
        self.close()
        if not self._bound and self._sim is not None and SimulationContext.instance() is self._sim:
            self._sim.clear_instance()

    def _retain_native_pair(self) -> None:
        """Keep the leased native pair alive while the simulation tears down around it.

        The session calls this when its retirement fails or is refused while the lease is still held, so
        a live or unproved render owner may still use the pair. The simulation then closes the
        renderer's native engine and the ``OvstageBackend`` that own it. Moving the native renderer out of
        the former and the Stage's resources out of the latter leaves both ``close()`` methods nothing to
        destroy; a module-level list holds them instead, so they also outlive this bootstrap and its
        session. Once detached, the pair no longer validates as the renderer's live pair, so the render
        owner cannot use it again and releases nothing; it stays alive until the process exits.
        """
        if self._retained_pair or self._renderer is None:
            return
        backend, scene = self._renderer.backend, self._renderer.scene
        retained: list[Any] = []
        if backend is not None and backend.renderer is not None:
            # The engine's close() destroys only the renderer it references; detach() needs ``attached``.
            retained.append(backend.renderer)
            backend.renderer = None
            backend.attached = False
        if scene is not None and scene is not backend:
            # OvstageBackend.close() releases the Stage, its paths and their dependent callbacks
            # (including the engine's detach) through ``resources``.
            retained.extend((scene.resources, scene.stage, scene.paths))
            scene.resources = contextlib.ExitStack()
        self._retained_pair = tuple(retained)
        if self._retained_pair:
            _RETAINED_NATIVE_PAIRS.append(self._retained_pair)
            logger.error(
                "OVXR retirement was not proved: keeping the leased OVRTX renderer and OVStage alive until the"
                " process exits. They cannot be reused; recreate the session in a new process."
            )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("OVXR bootstrap is closed")

    def _restore_terminal_mode_sentinel(self) -> None:
        if self._installed_terminal_mode_sentinel:
            if hasattr(builtins, _TERMINAL_MODE_SENTINEL):
                delattr(builtins, _TERMINAL_MODE_SENTINEL)
            self._installed_terminal_mode_sentinel = False
