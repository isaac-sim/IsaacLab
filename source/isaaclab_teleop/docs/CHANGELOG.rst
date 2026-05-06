Changelog
---------

0.3.9 (2026-04-29)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed installation failure on Windows by adding ``platform_system == 'Linux'``
  marker to the ``isaacteleop`` dependency, which is only available on Linux.


0.3.8 (2026-04-24)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Switched :class:`~isaaclab_teleop.xr_anchor_utils.XrAnchorSynchronizer` to import
  ``get_current_stage`` from :mod:`isaaclab.sim.utils.stage` instead of
  ``isaacsim.core.experimental.utils.stage``, aligning with the Isaac Lab API.


0.3.7 (2026-04-22)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated XR anchor prim creation to use :func:`isaaclab.sim.utils.prims.create_prim`
  instead of ``isaacsim.core.experimental.prims.XformPrim``.


0.3.6 (2026-04-21)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :attr:`~isaaclab_teleop.IsaacTeleopCfg.control_channel_uuid` for
  receiving teleop control commands (start/stop/reset) from the headset via
  an OpenXR message channel.  The channel is managed by TeleopCore's native
  ``teleop_control_pipeline`` mechanism.

* Added :class:`~isaaclab_teleop.teleop_message_processor.TeleopMessageProcessor`
  retargeter that converts raw message-channel payloads into boolean control
  signals for :class:`~isaacteleop.teleop_session_manager.DefaultTeleopStateManager`.

* Added :func:`~isaaclab_teleop.poll_control_events` helper,
  :class:`~isaaclab_teleop.ControlEvents` dataclass, and
  :class:`~isaaclab_teleop.SupportsControlEvents` protocol for polling
  start/stop/reset signals from any teleop device in a single call.

* Added :attr:`~isaaclab_teleop.IsaacTeleopDevice.last_control_events`
  property exposing the most recent control events from the message channel.
  Control events are automatically bridged to legacy
  :meth:`~isaaclab_teleop.IsaacTeleopDevice.add_callback` callbacks.

Changed
^^^^^^^

* :meth:`~isaaclab_teleop.IsaacTeleopDevice.reset` now injects a
  ``reset`` :class:`ExecutionEvents` into TeleopCore's ``ComputeContext``
  on the next pipeline step, resetting retargeter cross-step state.
  Previously only the XR anchor was reset.

Fixed
^^^^^

* Fixed ``record_demos.py`` not resetting the teleop device when a
  success condition triggers an environment reset.  Retargeters now
  reinitialize their state on success-triggered resets.

* Fixed shutdown hang caused by Kit's pre-shutdown callback calling
  ``stop()`` while the simulation loop was still running.  The callback
  now uses the same graceful teardown path as the XR-disabled handler.

0.3.5 (2026-04-06)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``cloudxr_env_file`` and ``auto_launch_cloudxr`` parameters to
  :func:`~isaaclab_teleop.create_isaac_teleop_device`,
  :class:`~isaaclab_teleop.IsaacTeleopDevice`, and
  :class:`~isaaclab_teleop.session_lifecycle.TeleopSessionLifecycle` for
  auto-launching the CloudXR runtime and WSS proxy during session startup.
  When a ``.env`` file path is provided via ``--cloudxr_env``, users no
  longer need to run ``python -m isaacteleop.cloudxr`` in a separate
  terminal.
* Added device-specific CloudXR ``.env`` profiles:
  :data:`~isaaclab_teleop.CLOUDXR_JS_ENV` (Quest/Pico, ``auto-webrtc``) and
  :data:`~isaaclab_teleop.CLOUDXR_AVP_ENV` (Apple Vision Pro, ``auto-native``).
* Added ``dex-retargeting==0.5.0`` as a required dependency on Linux x86_64.

Changed
^^^^^^^

* Made ``isaacteleop[retargeters,ui,cloudxr]~=1.2.0`` a required dependency of
  ``isaaclab_teleop`` (previously an optional extra via
  ``isaaclab_teleop[teleop]``).


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
