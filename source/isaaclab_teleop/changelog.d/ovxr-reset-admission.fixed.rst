* Gated live and MCAP replay input on admission of the current reset scene, including
  the initial replay publication. Automatic environment resets now invalidate the
  anchor and hand/action state before input resumes, without issuing another reset.
* Added :meth:`~isaaclab_teleop.IsaacTeleopDevice.notify_reset` so host-driven resets
  reset retargeters while preserving operator control events and withholding stale actions.
* Kept rejected action proposals from changing nested conditioner state, restricted
  hard replay resets to Kit, and rejected unsupported kitless HDF5 recording early.
