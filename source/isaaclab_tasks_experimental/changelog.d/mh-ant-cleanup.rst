Changed
^^^^^^^

* Changed the warp locomotion environment to resolve ``joint_gears`` by joint name expression,
  matching the stable ``LocomotionDirectEnv``. This drops the Newton-only restriction that the
  previous backend-keyed lookup imposed.
* Changed the warp locomotion environment to reject a configuration whose ``observation_space`` does
  not match the layout its observation kernel writes. The stable direct ant and humanoid tasks now
  observe feet joint wrenches and scale their rewards by ``step_dt``; the warp frontend has not been
  ported to either, so it raises instead of silently zero-filling the tail of the observation buffer.
