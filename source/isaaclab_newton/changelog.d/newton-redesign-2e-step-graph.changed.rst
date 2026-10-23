* Changed :meth:`~isaaclab_newton.physics.NewtonManager.step` to launch one :class:`~isaaclab.utils.StepGraph`
  covering its physics steps: collision, control callbacks, Newton actuators, solver substeps, post-step callbacks and
  sensors. Consecutive graphable operations replay from one CUDA graph; work that cannot be captured, such as
  TorchScript actuators or eager callbacks, runs in place between captured segments.
* **Breaking:** Removed ``NewtonManager.activate_newton_actuator_path``, ``register_pre_actuator_callback`` and
  ``unregister_post_step_callback``; use ``activate_actuators``, ``register_control_callback`` and
  ``unregister_step_callback`` with the handle returned by ``register_post_step_callback``.
