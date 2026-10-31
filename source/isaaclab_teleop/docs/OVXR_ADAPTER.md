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
shutdown. Importing the adapter requires the compatible Capture OVXR packages.

A failed scene write or seal stops further publication. Recreate the session
after retiring the failed pair; retrying the same generation cannot return an
older scene as if the failed update succeeded. During shutdown, publication
storage drains before lease guards are removed. The adapter retains the
environment if lease retirement cannot establish completion.

## Backend implementation

The ordinary renderer remains in `isaaclab_ov.renderers.ovrtx_renderer`.
`render_lease.RenderLeaseController` owns the lease state, thread-transfer
capability and per-owner operation guards; it borrows the renderer's native resources
and delegates component writes back to that renderer. `publication_buffers`
retains producer storage until native writes complete. Public renderer and lease
imports remain available from `isaaclab_ov.renderers`.

Acquiring a lease selects a private rendering OVStage without changing process
environment variables. An explicit OVStage disable or shared-stage selection is
rejected. Without an active lease, the controller permits
ordinary renderer operations and does not start an XR session. While a consumer
holds the lease, ordinary camera rendering, scene writes and renderer destruction
are rejected to prevent concurrent access to the borrowed resources. The same
generic borrowing interface is available to applications outside teleoperation.
