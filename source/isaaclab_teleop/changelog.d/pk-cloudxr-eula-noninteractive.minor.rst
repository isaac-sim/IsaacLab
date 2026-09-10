Added
^^^^^

* Added the ``ISAACLAB_CXR_ACCEPT_EULA=1`` environment variable, which accepts the NVIDIA
  CloudXR license up front wherever Isaac Lab launches the CloudXR runtime -- both the teleop
  session lifecycle and the process-scoped launcher in ``teleop_replay_agent.py``, which share
  one :func:`~isaaclab_teleop.cloudxr_eula_accepted` helper. The license is separate from the
  Omniverse one and was otherwise only ever prompted for on stdin, so headless, container and
  CI runs aborted with
  ``RuntimeError: CloudXR EULA was not accepted; cannot start the runtime``.
