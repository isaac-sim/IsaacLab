Changelog
---------

0.5.2 (2026-06-02)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed ``teleop_replay_agent.py``'s ``cpu_frame_time_ms`` and ``fps``
  percentiles, which previously projected per-``env.step`` CPU samples
  onto per-render units by dividing by ``decimation / render_interval``.
  Because each ``env.step`` folds multiple physics substeps and rendered
  frames into a single measurement, those sums are CLT-smoothed and
  underreport per-frame hitches the headset wearer / spectator actually
  sees. The agent now wraps
  :meth:`~isaaclab.sim.SimulationContext.render` and records the
  wall-clock interval between successive calls produced from inside
  ``env.step`` during the active window; that per-rendered-frame series
  is the new source for ``cpu_frame_time_ms`` and ``fps``. The run
  dict's ``active_iterations`` field is backed by a dedicated counter.
  Each interval is the wall-clock delta between successive
  ``env.sim.render`` calls, so at least two calls are required per
  run; runs that stepped the env but produced 0 or 1 renders during
  the active window raise ``RuntimeError`` from
  ``_run_single_replay`` (the agent aborts the batch without writing
  a stdout summary or JSON report, so "no JSON output" is an
  unambiguous measurement-failure signal for CI).
* Fixed the shipped CloudXR ``.env`` profiles to disable pose wait by default,
  preventing CloudXR frame pacing from throttling teleoperation sessions after
  frame-time spikes.


0.5.1 (2026-05-22)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added an ``env_cfg`` block to ``teleop_replay_agent.py``'s stats output
  capturing the performance- and frame-timing-relevant env config inputs
  (``sim.dt``, ``sim.render_interval``, ``decimation``, ``episode_length_s``,
  ``scene.num_envs``, ``sim.device``, ``sim.use_fabric``,
  ``sim.render.antialiasing_mode``) along with precomputed ``policy_dt_s``,
  ``render_dt_s``, ``renders_per_step``, ``target_policy_hz``, and
  ``target_render_hz`` rates. The same fields are echoed in a compact
  ``Env timing:`` line in the stdout summary so the measured
  ``cpu_frame_time_ms`` / ``fps`` numbers are self-interpreting across
  machines and configs without cross-referencing the env definition.

Changed
^^^^^^^

* Changed ``teleop_replay_agent.py``'s ``cpu_frame_time_ms`` and ``fps``
  blocks (both per-run and aggregate) to report on a **per-render** basis
  rather than per-``env.step``: each captured ``env.step`` CPU sample is
  divided by ``decimation / render_interval`` (the number of Kit renders
  per ``env.step``) before stats are computed. ``cpu_frame_time_ms.mean``
  now reads as the wall time between rendered frames and ``fps.mean``
  reads as the render rate -- the same number Kit's HUD shows, which is
  what the headset wearer / spectator actually perceives during real-time
  teleop. Field shapes and ``schema_version`` are unchanged. Falls back
  to the raw per-``env.step`` units when ``decimation`` or
  ``render_interval`` are unavailable from the env config.


0.5.0 (2026-05-20)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added MCAP record/replay support to :class:`~isaaclab_teleop.IsaacTeleopDevice` via new
  ``mcap_record_path`` and ``mcap_replay_path`` parameters on
  :func:`~isaaclab_teleop.create_isaac_teleop_device` (mutually exclusive). ``mcap_replay_path``
  switches the underlying :class:`isacteleop.teleop_session_manager.TeleopSession` into
  :class:`SessionMode.REPLAY` and feeds the recorded tracker stream through the configured
  retargeting pipeline; ``mcap_record_path`` is a debug-grade knob that writes the live session
  to a single continuous MCAP file for pairing with the replay agent in CI. It is **not** a
  data-generation format -- the produced MCAP has no per-episode segmentation, no world-frame
  anchor state, no env reset state, and no public Python decoder.
* Added a ``--mcap_record_path`` (debug-only) flag to ``scripts/tools/record_demos.py`` that
  forwards into :func:`~isaaclab_teleop.create_isaac_teleop_device` when the IsaacTeleop stack
  is in use.
* Added ``scripts/environments/teleoperation/teleop_replay_agent.py``, a non-interactive entry
  point used by CI to replay captured Isaac Teleop sessions against an Isaac Lab environment.
  The agent gates env stepping on :func:`~isaaclab_teleop.poll_control_events` so the recorded
  START / STOP / RESET boundaries reproduce the original recording's pacing, and asks Kit to
  ``post_quit`` on the first STOP-edge after teleop has been active so the host process exits
  deterministically.

Changed
^^^^^^^

* **Breaking:** Removed the ``isaaclab_teleop.automation`` subpackage, including
  ``XcrReplayConfig`` and ``start_xcr_replay``. The XCR backend was a transitional Kit-level
  OpenXR capture/replay path that pre-dated Isaac Teleop's native MCAP record/replay. Replays
  now go through ``teleop_replay_agent.py`` against an MCAP capture produced by Isaac Teleop.
* **Breaking:** Removed the lazy legacy ``teleop_devices`` (``handtracking`` / ``manusvive``)
  accessor on
  :class:`~isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_gr1t2_env_cfg.PickPlaceGR1T2EnvCfg`.
  All in-tree scripts (``teleop_se3_agent.py``, ``record_demos.py``, ``teleop_replay_agent.py``)
  prefer ``env_cfg.isaac_teleop``; consumers that built the legacy
  :class:`~isaaclab.devices.openxr.OpenXRDevice` directly from the env config should construct
  it themselves or migrate to :class:`~isaaclab_teleop.IsaacTeleopDevice`.


0.4.0 (2026-05-16)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``scripts/environments/teleoperation/teleop_replay_agent.py``, a
  non-interactive entry point used by CI to replay captured teleop sessions
  against an Isaac Lab environment, plus a small internal
  ``isaaclab_teleop.automation`` subpackage backing it. Replaces the runtime
  patch the ``teleop-cicd`` pipeline previously applied to
  ``teleop_se3_agent.py``.
* Expanded the **Optimize XR Performance** documentation with guidance for
  lower-spec GPUs and complex scenes: a walkthrough for switching the
  Isaac Lab viewport to the RTX - Minimal renderer (including the
  ``DistantLight``-only lighting limitation), notes on the
  ``sim.dt`` / ``sim.render_interval`` trade-off, a description of the
  XR **Resolution Multiplier** slider for trading image sharpness for GPU
  headroom, guidance on ``RetargetingExecutionConfig`` (sync vs pipelined
  modes and ``DeadlinePacingConfig.safety_margin_s``), and a CloudXR
  frame-pacing diagnostic note. See :ref:`isaac-teleop-performance`.

Changed
^^^^^^^

* Added :paramref:`~isaaclab_teleop.automation.XcrReplayConfig.max_replay_duration_s`
  (default: ``3600``) so the completion-poll loop in
  :func:`~isaaclab_teleop.automation.start_xcr_replay` is bounded. If
  Kit's :mod:`xcr_player` ever fails to clear its private playback
  subscription, the coroutine now returns instead of spinning forever.
* Stored the :class:`omni.kit.xr.core.recorder._xr_xcr.XCRReplayAPI`
  instance in a local variable inside
  :func:`~isaaclab_teleop.automation.start_xcr_replay` so it stays alive
  for the lifetime of the replay coroutine.

Fixed
^^^^^

* Fixed ``teleop_replay_agent.py`` driving the robot toward the world origin
  for the duration of ``--replay_start_delay_s``. The legacy
  :class:`~isaaclab.devices.openxr.OpenXRDevice` returns a default zero pose
  while the OpenXR runtime is silent, so calling ``env.step()`` during the
  start-delay window fed the Pink IK garbage targets and corrupted the robot
  pose long before real hand-tracking data flowed. The agent now registers
  ``"START"`` / ``"STOP"`` callbacks on the device -- the same path
  ``record_demos.py`` uses -- and only steps the env once the XCR replay
  dispatches the recorded ``"start"`` message through Kit's OpenXR message
  bus.
* Fixed ``teleop_replay_agent.py`` hanging the CI process when the XCR
  replay driver coroutine raised before reaching ``post_quit``. The
  previously discarded :class:`asyncio.Future` is now retained and a done
  callback logs the failure with traceback and asks Kit to quit so the
  host process exits cleanly.
* Fixed ``teleop_replay_agent.py`` leaking the USD stage when device
  construction or environment setup raised. ``env.close()`` now runs from a
  ``try/finally`` block so cleanup happens on every exit path.
* Fixed ``teleop_replay_agent.py`` producing a frozen-arms / hands-only
  symptom during replay. Kit's ``teleop_command`` message bus drains
  queued events as a batch when the AR profile is enabled, so the
  recorded user's STOP gesture would fire within milliseconds of START
  and gate ``env.step()`` off again before Pink IK had time to converge.
  The replay agent now subscribes only to ``"START"``: replay is one-shot
  and the only valid termination is the driver's ``post_quit``.
* Aligned ``teleop_replay_agent.py``'s pre-loop reset sequence with
  ``record_demos.py`` -- ``env.sim.reset()`` then ``env.reset()`` then
  ``teleop_interface.reset()`` -- so the hard physics reinit re-binds the
  articulation tensor views that
  :meth:`~isaaclab.controllers.pink_ik.PinkIKController.compute` reads
  from each step.
* Cleared :attr:`~isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_gr1t2_env_cfg.TerminationsCfg.success`
  in the replay env config so a successful replay does not snap the robot
  back to its initial pose mid-loop.


0.3.11 (2026-05-12)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :attr:`~isaaclab_teleop.IsaacTeleopCfg.retargeting_execution` for
  configuring IsaacTeleop retargeting execution mode from Isaac Lab.

Changed
^^^^^^^

* Changed :class:`~isaaclab_teleop.IsaacTeleopCfg` to enable IsaacTeleop
  deadline-paced pipelined retargeting by default. This returns the latest
  completed retargeting output while the current frame is submitted, using
  ``DeadlinePacingConfig(safety_margin_s=0.025)`` to sample close to the next
  simulation consumption point and stagger IsaacTeleop's Python work behind
  Isaac Lab's step Python. Set
  ``retargeting_execution=RetargetingExecutionConfig(mode="sync")`` to restore
  exact current-frame retargeting.

Fixed
^^^^^

* Fixed installation to upgrade to the latest compatible ``isaacteleop``
  package when installing ``isaaclab_teleop``.


0.3.10 (2026-05-08)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed ``--teleop_device`` default to ``None`` in ``teleop_se3_agent.py``
  and ``record_demos.py``. When omitted, the IsaacTeleop pipeline is used if
  the env configures ``isaac_teleop``; otherwise keyboard is used as fallback.
  When explicitly provided, the scripts use the legacy ``teleop_devices`` path
  and error out if no matching entry exists.
* Removed automatic ``--xr`` detection from ``--teleop_device`` containing
  ``"handtracking"``. Users who need XR with the legacy path should pass
  ``--xr`` explicitly.


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
