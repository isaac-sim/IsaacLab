Added
^^^^^

* Added the ``ISAACLAB_CXR_ACCEPT_EULA=1`` environment variable, which accepts the NVIDIA
  CloudXR license up front when the teleop scripts auto-launch the CloudXR runtime. The
  license is separate from the Omniverse one and was otherwise only ever prompted for on
  stdin, so headless, container and CI runs aborted with
  ``RuntimeError: CloudXR EULA was not accepted; cannot start the runtime``.
