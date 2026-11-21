* Changed ``run_teleoperation`` to own its cleanup: it no longer takes a ``cleanup`` stack, and
  requires the keyword-only ``use_isaac_teleop`` argument.
* Changed the live teleoperation agent and the demonstration recorder, also with Kit, to report
  environment resets to the input pipeline with :meth:`~isaaclab_teleop.IsaacTeleopDevice.notify_reset`:
  after an operator reset, and when environment 0 terminates or is truncated during a step. On such a
  step the live agent resets a keyboard, gamepad or SpaceMouse device instead.
* Changed MCAP replay, also with Kit, to end a run as a failure when a retargeting step fails. It
  restarted the input session before, which replayed the recording from its start.
* Changed the MCAP replay agent to rebuild the simulation with ``env.sim.reset()`` before a run and
  after a recorded RESET only for Isaac Sim PhysX. Newton and OvPhysX keep their simulation, also with
  Kit.
* Changed the MCAP replay agent to raise when the CloudXR runtime it launched fails to stop. It logged
  that failure before; no report is written now.
