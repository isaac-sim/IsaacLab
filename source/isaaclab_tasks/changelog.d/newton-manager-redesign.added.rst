* Added ``IsaacContrib-NewtonStep-*`` tasks that exercise work inside the Newton step: Franka reach with the Newton
  operational-space controller as a control callback, and flat ANYmal-D with a sensor suite, DC-motor actuators, or
  mixed DC-motor and LSTM actuators.
* Added :class:`~isaaclab_tasks.contrib.newton_step.captured_cartpole.CapturedCartpole`, ``Isaac-Cartpole``'s MDP as
  Warp kernels whose whole environment step, Newton physics included, replays as one CUDA graph.
