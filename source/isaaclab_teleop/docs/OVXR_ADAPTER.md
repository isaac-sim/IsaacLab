# OVXR application adapter

`isaaclab_teleop.OvLibrariesXrSession` connects an Isaac Lab environment to
Isaac Capture's OVXR session owner. It uses the simulation's existing OVRTX
renderer and OVStage through `isaaclab_ov.renderers.OvLibrariesRenderLease`.

## Ownership

- `isaaclab_ov` owns the renderer, stage, scene publication and their lifetime.
  Its lease validates access and enforces render-thread ownership. The adapter's
  publication guard serializes application writes with Capture's frame operations.
  It does not create an XR session or select teleoperation behavior.
- `isaaclab_teleop.ovxr_session.OvxrSessionBootstrap` creates the simulation
  context and reserves the lease before environment construction. It binds the
  completed environment and reads current target-frame transforms.
- `isaaclab_teleop.ov_libraries_xr.OvLibrariesXrSession` supplies the application
  adapter to Isaac Capture. It declares XR render products, publishes scene and
  anchor updates, transfers render-thread ownership and coordinates shutdown.
- Isaac Capture owns the OVXR session and render worker. Its borrowed OpenXR
  handles are available to the teleoperation input session.

## Lifecycle

Prepare the adapter before constructing the environment so that both use the
same simulation context and native renderer/scene pair. Start the render worker
after the environment exists. Publish simulation updates through the session;
the worker owns rendering. Stop input consumers before closing the XR session,
and retire the lease before destroying the simulation's renderer or stage.
If startup fails before the worker takes ownership, the caller closes the
environment after releasing the lease. If input retirement fails, request
shutdown and retain the borrowed resources until their retirement is proved.

The lease currently requires one environment and a private rendering OVStage.
Ownership transfer is single-use. Failed retirement must preserve resources
for the owning thread to finish cleanup; a failed drain is not successful
shutdown. Importing `OvLibrariesXrSession` requires a Capture release with OVXR
session support; see [Installing kitless XR](kitless_xr_source.md). The
`OvLibrariesXrCfg` dataclass lives in `isaaclab_teleop.ovxr_cfg` and imports
without those packages.

A failed scene write or seal stops further publication. Recreate the session
after retiring the failed pair; retrying the same generation cannot return an
older scene as if the failed update succeeded. The failure poisons producer
publication only: the render-owner lane stays available so Capture can destroy
its OVXR context and the adapter can release the lease and close the
environment. After a poisoned lease is released, the renderer keeps rejecting
ordinary use (except closing it) because its write ordinal was not rebased.
During shutdown, publication storage drains before lease guards are removed.
The adapter retains the environment if lease retirement cannot establish completion.

## Environment requirements

- `OvLibrariesXrSession.prepare()` creates the `SimulationContext` from the
  `sim_cfg` it receives, and the environment reuses that context. Pass the
  environment's own `env_cfg.sim`; before the worker claims the lease,
  `start_render_worker()` rejects an environment whose simulation device or
  time step differs, or whose `num_envs` is not 1.
- Only environments that accept a pre-created simulation context are
  supported. Manager-based environments do; `DirectRLEnv` always creates its own
  context and fails with "Simulation context already exists".
- Camera sensors on other renderers are supported: their initialization
  prepares every registered renderer on the application thread, and the leased
  renderer defers its Stage until the XR worker claims ownership. A camera that
  resolves to the leased renderer itself (for example an OVRTX camera with the
  default `OVRTXRendererCfg()`) is rejected while the lease is active.
- Eager scene updates (`lazy_sensor_update=False`) skip the leased renderer; its
  scene publisher writes the same physics state at the reserved ordinal.
- CPU simulation devices publish scene transforms, but visual-material
  publication requires CUDA material buffers, as it does for ordinary OVRTX
  cameras. With `--device cpu`, an environment that has `VisualMaterial` assets
  fails at `sim.reset()`.
- Dynamic anchoring takes any object implementing
  `OvLibrariesXrSession.ExternalAnchorState`; `start_render_worker()` rejects
  other objects before the worker claims the lease.

## Backend implementation

The ordinary renderer remains in `isaaclab_ov.renderers.ovrtx_renderer`.
`render_lease.RenderLeaseController` owns the lease state, thread-transfer
capability and per-owner operation guards; it borrows the renderer's native resources
and delegates component writes back to that renderer. Public renderer and lease
imports remain available from `isaaclab_ov.renderers`.

Acquiring a lease selects a private rendering OVStage without changing process
environment variables. An explicit OVStage disable or shared-stage selection is
rejected. Without an active lease, the controller permits
ordinary renderer operations and does not start an XR session. While a consumer
holds the lease, ordinary camera rendering, scene writes and renderer destruction
are rejected to prevent concurrent access to the borrowed resources. The same
generic borrowing interface is available to applications outside teleoperation.

## Native validation

The adapter and lease unit tests double the native OVRTX and OVStage endpoints.
An opt-in harness runs the render-owner transfer and complete scene publication
against the real libraries, on the manager-based `Isaac-Cartpole` task with its
`ovphysx` physics preset and one environment on `cuda:0`. The transfer case
also runs with CPU physics, which publishes host scene tensors to the GPU
renderer. It needs a CUDA GPU, OVRTX, OVStage, OvPhysX and an Isaac Capture
release with OVXR session support; it starts no XR session or CloudXR runtime.
Each case runs in a child process:

```bash
ISAACLAB_RUN_NATIVE_OVRTX_TRANSFER=1 uv run python -m pytest \
    source/isaaclab_teleop/test/test_ovrtx_transfer_native_harness.py
```

Without `ISAACLAB_RUN_NATIVE_OVRTX_TRANSFER=1`, every case is skipped.
