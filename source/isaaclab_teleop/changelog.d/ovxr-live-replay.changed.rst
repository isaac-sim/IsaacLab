* **Breaking:** Replay follows recorded START and RESET controls by default. Add
  ``--start_on_replay`` for recordings made without START events that should begin
  immediately.
* **Breaking:** Teleoperation context exit reports session or runtime cleanup failures
  to the caller instead of silently suppressing them. Handle the reported failure;
  resources whose asynchronous users have not retired must remain alive until
  cleanup succeeds.
* **Breaking:** Live teleoperation now exits non-zero when environment creation, device
  creation or a teleoperation step fails, instead of logging the error and exiting 0.
  Automation that inspected the log for failures should check the exit code.
* Changed ``--disable_external_cameras`` to remove camera sensors in every live teleoperation
  mode, not only with ``--xr``.
* Changed replay to launch Kit only when the resolved task configuration requires it (for
  example Isaac Sim PhysX, ``--xr`` or a Kit visualizer) instead of always. Select a Kit
  physics preset to keep the previous runtime for a kitless task.
