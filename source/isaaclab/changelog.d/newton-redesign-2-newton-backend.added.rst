* Added :class:`~isaaclab.utils.StepGraph` and :class:`~isaaclab.utils.StepOp`, which run a step as straight-line
  operations grouped into CUDA graphs with eager islands for work that cannot be captured.
* Added :meth:`~isaaclab.actuators.newton.NewtonActuatorAdapter.zero_outputs`,
  :meth:`~isaaclab.actuators.newton.NewtonActuatorAdapter.reset_dofs` and the ``state_buffers`` and
  ``computed_effort`` properties, so a physics step graph can run Newton actuators over a model's flat DOF space.
