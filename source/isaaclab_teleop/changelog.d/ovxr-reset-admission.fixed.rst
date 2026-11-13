* Gated live and MCAP replay input on admission of the current reset scene, including
  the initial replay publication. Automatic environment resets now invalidate the
  tracking anchor before input resumes, without issuing another environment reset.
* Added :meth:`~isaaclab_teleop.IsaacTeleopDevice.notify_reset` so host-driven resets
  reset retargeters with fresh external inputs while preserving operator control events
  and withholding stale actions. Hard replay resets remain limited to Kit.
