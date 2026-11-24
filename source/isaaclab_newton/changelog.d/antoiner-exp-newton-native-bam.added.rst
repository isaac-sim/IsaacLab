* Added :func:`~isaaclab_newton.actuators.get_bam_group` and :class:`~isaaclab_newton.actuators.BamGroup`
  to read and write a Newton-executed BAM group's friction scale and its batteries' nominal voltage and
  resistance, and to read its friction budget.
* Added :attr:`~isaaclab_newton.physics.NewtonBackend.actuator_model_sync_flags` to have the solver re-read
  model properties, such as joint friction written by pre-actuator hooks, once per step after the actuators.
