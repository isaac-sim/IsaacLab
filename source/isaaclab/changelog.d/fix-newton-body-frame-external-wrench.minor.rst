Added
^^^^^

* Added :meth:`~isaaclab.utils.wrench_composer.WrenchComposer.compose_to_world_frame` together with
  the :attr:`~isaaclab.utils.wrench_composer.WrenchComposer.out_force_w` and
  :attr:`~isaaclab.utils.wrench_composer.WrenchComposer.out_torque_w` outputs, which express the
  composed external wrench in the world frame at the body's center of mass. This complements the
  existing body-frame composition for solvers that expect world-frame external wrenches.
