* Added :class:`~isaaclab_teleop.OvLibrariesXrSession`, an OVXR application adapter and session
  bootstrap that borrow the simulation-owned OVRTX renderer and OVStage, publish scene updates,
  and coordinate render-thread ownership and shutdown with Isaac Capture.
  Publication failures close admission, and failed lease retirement retains borrowed resources.
* Added :class:`~isaaclab_teleop.OvLibrariesXrCfg` in :mod:`isaaclab_teleop.ovxr_cfg`, which
  imports without the native OVXR packages.
