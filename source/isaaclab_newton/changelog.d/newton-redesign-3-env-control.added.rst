* Added :meth:`NewtonManager.handle_actions <isaaclab_newton.physics.NewtonManager.handle_actions>`. When every action
  term and written asset could be recorded, the environment's actions and asset writes ran as pre-actuator callbacks
  before every physics step, and one step advanced the whole decimation loop. Hard resets registered them again.
* Added ``NewtonBackend.handles_actions``. When the step handled actions, every physics step staged the body forces
  that the asset writes produced, and reordered or ball-jointed articulations republished their state before writing,
  so the writes read the current physics step.
