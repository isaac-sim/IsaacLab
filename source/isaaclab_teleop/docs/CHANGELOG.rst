Changelog
---------

0.3.4 (2026-03-17)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :attr:`~isaaclab_teleop.IsaacTeleopCfg.target_frame_prim_path` for
  config-driven frame rebasing.  When set to a USD prim path, the device
  automatically reads the prim's world transform each frame and uses its
  inverse as the ``target_T_world`` rebase matrix, so all output poses are
  expressed in the target frame (e.g. robot base link for IK).

* Added ``target_T_world`` parameter to
  :meth:`~isaaclab_teleop.IsaacTeleopDevice.advance` for rebasing all output
  poses into an arbitrary target coordinate frame (e.g. robot base link for
  IK).  Accepts :class:`numpy.ndarray`, :class:`torch.Tensor`, or
  ``wp.array``.


0.3.3 (2026-03-13)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed race condition in headless XR where ``xr.profile.ar.enabled`` was set
  in the ``.kit`` file before the teleop bridge extension finished loading,
  causing ``BridgeComponent`` to miss its lifecycle callbacks.  The setting is
  now deferred to
  :meth:`~isaaclab_teleop.session_lifecycle.TeleopSessionLifecycle._ensure_xr_ar_profile_enabled`
  after all extensions have loaded.


0.3.2 (2026-03-12)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Add nvidia-srl-usd-to-urdf dependency to isaaclab_teleop extension.


0.3.1 (2026-02-26)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Add cleanup for Isaac Teleop session when Stop XR button is clicked and when Kit is closed.


0.3.0 (2026-02-26)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Update Isaac Teleop API usage for querying controller button states.


0.2.0 (2026-02-24)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :meth:`~isaaclab_teleop.session_lifecycle.TeleopSessionLifecycle._on_request_required_extensions` to request required
  OpenXR extensions at runtime based on Teleop devices needed for the specified environment.

0.1.0 (2026-02-18)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Initial release of the ``isaaclab_teleop`` extension.

* Added :class:`~isaaclab_teleop.IsaacTeleopDevice` providing a unified teleoperation interface
  that manages IsaacTeleop sessions, XR anchor synchronization, and retargeting pipelines within
  Isaac Lab environments.

* Added :class:`~isaaclab_teleop.IsaacTeleopCfg` for pipeline-based configuration of
  retargeting, XR anchors, and device settings directly in environment configs.
