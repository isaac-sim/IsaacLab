# OVXR application adapter

`isaaclab_teleop.OvLibrariesXrSession` connects an Isaac Lab environment to
Isaac Capture's OVXR session owner. The simulation's scene reaches the session through
`isaaclab_teleop.ovxr_renderer.OvxrRenderer` (selected by `OvxrRendererCfg`), a renderer that
publishes the scene to its own OVStage and OVRTX engine, which the session borrows through its lease.

## Ownership

- The simulation owns the renderer and stage and their lifetime. `OvxrRenderer`'s lease
  validates access, enforces render-thread ownership and orders scene publication. The adapter's
  publication guard serializes application writes with Capture's frame operations.
  It does not create an XR session or select teleoperation behavior.
- `isaaclab_teleop.ovxr_session.OvxrSessionBootstrap` creates the simulation
  context and reserves the lease before environment construction. It binds the
  completed environment and reads current target-frame transforms.
- `isaaclab_teleop.ov_libraries_xr.OvLibrariesXrSession` supplies the application
  adapter to Isaac Capture. It declares the two eye cameras, publishes scene and
  anchor updates, transfers render-thread ownership and coordinates shutdown.
  OVXR authors its own eye render products, render vars and render settings.
- Isaac Capture owns the OVXR session and render worker. Its borrowed OpenXR
  handles are available to the teleoperation input session.

## Lifecycle

Prepare the adapter before constructing the environment so that both use the
same simulation context and native renderer/scene pair. Start the render worker
after the environment exists. Publish simulation updates through the session;
the worker owns rendering. Stop input consumers before closing the XR session,
then close the environment on the host thread.

`prepare()` registers the session's retirement as the simulation's first
`PhysicsEvent.STOP` listener. Closing the environment clears the kitless
simulation context, which dispatches STOP before Isaac Lab releases cameras,
renderers, the native OVRTX engine and the OVStage, so the worker drains and
the lease is released first. The listener closes a session that is still open
only if it handed out no OpenXR handles. A STOP while the simulation runs (a
hard reset or a physics re-warm) raises and leaves the session open.

If startup fails before the worker takes ownership, the caller closes the
environment after releasing the lease. Closing the session waits for a worker
that is still attaching the scene, because attachment cannot be cancelled and
reads the environment. If input retirement fails, request
shutdown and leave the session open. Whenever the STOP listener cannot prove
retirement during a shutdown, it detaches the leased renderer and OVStage from
the simulation's backends, keeps them alive until the process exits, logs an
error and raises, so teardown destroys neither under a live render owner.

The lease currently requires one environment, and the session renders on its own OVStage.
Ownership transfer is single-use. Failed retirement must preserve resources;
a failed drain is not successful shutdown. Importing `OvLibrariesXrSession` requires a Capture release with OVXR
session support; see [Installing kitless XR](kitless_xr_source.md). The
`OvLibrariesXrCfg` dataclass lives in `isaaclab_teleop.ovxr_cfg` and imports
without those packages.

A failed scene write or seal stops further publication. Recreate the session
after retiring the failed pair; retrying the same generation cannot return an
older scene as if the failed update succeeded. The failure poisons producer
publication only: the render-owner lane stays available so Capture can destroy
its OVXR context and the adapter can release the lease before the host closes the
environment. A released renderer cannot be leased again; only closing it remains.
During shutdown, publication storage drains before lease guards are removed.
The adapter retains the renderer and Stage if lease retirement cannot establish completion.

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
  prepares every registered renderer on the application thread, and the XR
  renderer defers its Stage until the XR worker claims ownership. A camera that
  resolves to the XR renderer itself (a non-panel camera configured with the
  session's `OvxrRendererCfg`) is rejected, because that renderer produces no camera images.
- Camera panels provide colour (`rgb`, `rgba`) or `depth` outputs. On a CPU simulation
  device, or when `OvxrRendererCfg.active_cuda_gpus` excludes the simulation GPU, their
  render products leave GPU selection to OVRTX.
- Eager scene updates (`lazy_sensor_update=False`) write nothing to the XR renderer;
  its scene publisher writes the same physics state at the reserved ordinal. Camera
  panels' sensors would still attempt their own capture, which the XR renderer rejects,
  so camera panels require `lazy_sensor_update=True`; the adapter rejects eager
  retained cameras before starting the worker. The scene publisher refreshes
  their world poses without triggering camera capture before publication.
- CPU simulation devices publish scene transforms, but visual-material
  publication requires CUDA material buffers, as it does for Isaac Lab's OVRTX
  renderer. With `--device cpu`, an environment that has `VisualMaterial` assets
  fails at `sim.reset()`.
- Dynamic anchoring takes any object implementing
  `OvLibrariesXrSession.ExternalAnchorState`; `start_render_worker()` rejects
  other objects before the worker claims the lease.

## Backend implementation

`isaaclab_teleop.ovxr_renderer.OvxrRenderer` implements Isaac Lab's `BaseRenderer` interface;
Isaac Lab's OVRTX renderer in `isaaclab_ov.renderers.ovrtx_renderer` has no XR knowledge. The
private module `isaaclab_teleop.ovxr_scene` holds the native engine, debug markers,
visual-material writes and camera-panel render data, following the OVStage path of the OVRTX
renderer. Both reuse `isaaclab_ov`'s stage holder (`OvstageBackend`), clone context,
render-product builder, transform kernels and tensor helpers.

- `OvxrRenderer` works only under the session's lease. It prepares its Stage only on the claimed
  render owner, ordinary scene updates write nothing, camera sensors other than the session's
  camera panels and all camera rendering are rejected, and its resources, including the panels'
  render products, cannot be released while the lease is active.
- Its `_OvxrLeaseController` owns the lease state, the thread-transfer capability, the
  per-owner operation guards and the publication order.
- Its `_OvxrPanelCameras` own the selected camera panels: matching their sensors, creating
  their products on the render owner once the Stage is populated, authoring native frame rates
  and selecting them in the Stage's population, and writing their pose samples with each scene.
- `OvxrScenePublisher.publish()` publishes each complete scene as one transaction: it skips an
  unchanged state, validates panel samples, reserves an ordinal, writes every component, seals
  it, and on failure poisons the pair and abandons the ordinal. Capture's publication guard
  provides the cross-thread exclusion around it.
- With `OvxrRendererCfg.active_cuda_gpus` excluding the physics GPU, rigid poses and points are
  copied to the first rendering GPU before OVStage consumes them
  (`isaaclab_teleop.ovxr_gpu_publication`).
- The OVXR engine (`isaaclab_teleop.ovxr_scene`) applies `active_cuda_gpus` and guards it. OVRTX
  selects rendering GPUs once per process, so the first OVXR engine fixes the selection and a later
  one that resolves to another selection is rejected, including after teardown or failed startup. A
  selection other than the simulation GPU alone is also rejected when an ordinary OVRTX renderer or
  engine is registered in the same simulation, when the engine is created and again when the Stage is
  prepared. Ordinary OVRTX renderers in another simulation of the same process are not checked.

The lease registers the renderer as a consumer of the simulation's visualization-marker registry.
This enables the existing marker backend even without an ordinary visualizer and lets lazily created
tracking markers participate in scene publication. The registration and registry subscription retire
with the lease.

On the session's presentation Stage, the bootstrap selects the `viewer` partition for the
eye cameras and disables all-partition viewing. Task geometry and environment lights,
including native copies, are common to the eye and sensor views; tracking markers remain
in `viewer`. Sensor images therefore exclude tracking markers without changing the
application USD or ordinary OVRTX rendering defaults.

The lease classes are internal to `isaaclab_teleop`.

Acquiring a lease reserves the renderer. After the session applies its renderer
settings, and before the environment is built, it creates its own rendering OVStage
and the OVRTX engine that attaches to it, through the simulation's backend registry.
Clone preparation delivers the scene copies to that Stage through the
`OvstageReplicateContext` that `OvxrRendererCfg` declares, and the simulation closes the
engine before the Stage. Without an active lease, the renderer prepares no Stage and
starts no XR session. While a consumer holds the lease, releasing the renderer's
resources is rejected to prevent concurrent access to the borrowed resources. Cameras
with an ordinary `OVRTXRendererCfg` keep Isaac Lab's OVRTX renderer and its own engine.

## Optional presentation

After resolving the selected camera render products, the Lab adapter supplies
Capture with configured `CameraPanelConsumer` and `OwnedDesktopMirror` instances
in `OvxrSessionBinding.frame_consumers`. Each panel configuration pairs its product
path with its placement and image bound. The mirror receives its camera choices
separately, so it can run without headset panels.

Capture runs both consumers through the same frame lifecycle. It shares CPU
snapshots within a frame and keeps native layers alive through submission; Lab
continues to own sensor production and scene publication. These interfaces are
application-neutral and do not require Lab-specific scene types.

The mirror uses OVUI's default Vulkan backend. Neither Lab nor Capture modifies
`OMNIUI_BACKEND`; the mirror rejects an active non-Vulkan backend. Neither optional
consumer is created when disabled.

## Native validation

The adapter and lease unit tests double the native OVRTX and OVStage endpoints.
An opt-in harness runs the render-owner transfer and complete scene publication
against the real libraries, on the manager-based `Isaac-Cartpole` task with its
`ovphysx` physics preset and one environment on `cuda:0`. The transfer case
also runs with CPU physics, which publishes host scene tensors to the GPU
renderer. Both cases also run `IsaacContrib-PickPlace-GR1T2-Abs` with CPU
OvPhysX, which its registered entry point prepares for kitless physics; its
cameras are removed as in an `--ovxr` launch. The harness needs a CUDA GPU,
OVRTX, OVStage, OvPhysX and an Isaac Capture release with OVXR session support;
it starts no XR session or CloudXR runtime.
Each case runs in a child process:

```bash
ISAACLAB_RUN_NATIVE_OVRTX_TRANSFER=1 uv run python -m pytest \
    source/isaaclab_teleop/test/test_ovrtx_transfer_native_harness.py
```

Without `ISAACLAB_RUN_NATIVE_OVRTX_TRANSFER=1`, every case is skipped.
