Changed
^^^^^^^

* Changed task-space action docstrings that advised keeping gravity compensation
  disabled on the Newton backend — :attr:`~isaaclab.assets.BaseArticulationData.gravity_compensation_forces`
  is now implemented on Newton, so ``gravity_compensation`` /
  ``enable_gravity_compensation`` may be enabled on both backends.
