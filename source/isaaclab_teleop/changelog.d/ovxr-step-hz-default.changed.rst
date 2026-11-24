* Changed ``--ovxr_step_hz`` in the live teleoperation agent and the demonstration recorder to default to
  the environment step rate, ``1 / (sim.dt * decimation)``, instead of leaving active control unthrottled.
  Simulation time targets real time, so recorded demonstrations keep the operator's pace. Pass
  ``--ovxr_step_hz 0`` for the previous unthrottled behavior.
