* **Breaking:** Replay follows recorded START and RESET controls by default. Add
  ``--start_on_replay`` for recordings made without START events that should begin
  immediately.
* **Breaking:** Teleoperation context exit reports session or runtime cleanup failures
  to the caller instead of silently suppressing them. Handle the reported failure;
  resources whose asynchronous users have not retired must remain alive until
  cleanup succeeds.
