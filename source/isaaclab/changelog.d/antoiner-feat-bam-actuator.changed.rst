* **Breaking:** Updated Newton-native actuators to the Newton 1.6 drive API.
  :func:`~isaaclab.actuators.newton.read_group_parameter` and
  :func:`~isaaclab.actuators.newton.write_group_parameter` took ``"drive"`` instead of ``"controller"``,
  and raw Newton access used ``actuator.drive`` instead of ``actuator.controller``.
  Migration: replace ``"controller"`` with ``"drive"`` in group parameter calls.
