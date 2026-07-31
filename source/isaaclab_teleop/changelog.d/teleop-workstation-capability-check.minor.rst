Added
^^^^^

* Added :func:`~isaaclab_teleop.check_system_requirements`, which measures the workstation against
  the recommended teleoperation spec (CPU single-thread throughput, frequency governor, GPU memory
  and architecture, driver, and system memory) and reports any unmet requirement. The check runs
  automatically when a teleop session starts, is advisory only, and never blocks a session.

* Added :meth:`~isaaclab_teleop.IsaacTeleopDevice.send_client_message` for sending JSON messages
  from Isaac Lab to the connected XR client over the teleop control channel. Workstation warnings
  use it to surface in the headset, where the operator can see them.
