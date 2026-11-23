* Changed ``--ovxr_step_hz`` in the live teleoperation agent and the demonstration recorder to default to
  the environment step rate, ``1 / (sim.dt * decimation)``, instead of leaving active control unthrottled.
  Simulation time now passes at real time when the host keeps up, so recorded demonstrations keep the
  operator's pace. Pass ``--ovxr_step_hz 0`` for the previous unthrottled behavior. The option is also
  rejected without ``--ovxr`` when its value is zero.
