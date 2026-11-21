* Added :class:`~isaaclab_teleop.OvLibrariesXrSession`, an OVXR application adapter and session
  bootstrap that borrow the simulation-owned OVRTX engine and OVStage, publish scene updates,
  and coordinate render-thread ownership and shutdown with Isaac Capture.
  A failed scene publication stops further publication, and a session whose renderer cannot be
  retired keeps its borrowed resources.
* Added :class:`~isaaclab_teleop.OvxrRendererCfg`, which selects the OVXR session's renderer, a
  :class:`~isaaclab.renderers.BaseRenderer` that publishes the simulation's scene to its own OVStage
  and OVRTX engine on CUDA and CPU simulation devices. It lends them to Isaac Capture's render worker,
  with a single-use render-thread ownership transfer, deferred Stage preparation, ordered scene
  publication (including dirty visual materials) and ordered cleanup. It renders no camera sensors;
  cameras keep their own renderers.
* Added :class:`~isaaclab_teleop.OvLibrariesXrCfg` in :mod:`isaaclab_teleop.ovxr_cfg`, which
  imports without the native OVXR packages.
* Added tracking and debug-visualization markers to the OVXR eye views; they do not need an ordinary
  visualizer.
