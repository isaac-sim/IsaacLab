Changed
^^^^^^^

* Changed task-space action docstrings that advised keeping gravity compensation
  disabled on the Newton backend — :attr:`~isaaclab.assets.BaseArticulationData.gravity_compensation_forces`
  is implemented on Newton builds that provide the inverse-dynamics API (newton > 1.3),
  so ``gravity_compensation`` / ``enable_gravity_compensation`` may be enabled there.
