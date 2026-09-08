Changed
^^^^^^^

* Changed the warp locomotion environment to resolve ``joint_gears`` by joint name expression,
  matching the stable ``LocomotionDirectEnv``. This drops the Newton-only restriction that the
  previous backend-keyed lookup imposed. Joints the table does not match keep a unit gear, as they do
  in the manager-based action and reward terms.
* Changed the warp locomotion environment to implement the same MDP as the stable direct and
  manager-based ant and humanoid tasks, so ``--frontend warp`` trains against the same problem. It
  now observes the feet joint wrenches, randomizes the joint state on reset, scales the continuous
  reward terms by the environment step interval, weighs the energy and joint-limit penalties by the
  per-joint gear ratio, and applies the death cost as a one-off terminal penalty.

Fixed
^^^^^

* Fixed the warp locomotion environment not logging ``Metrics/success_rate``, which both stable
  workflows report. The rate is now reduced on device and exposed as a tensor view, so the
  computation stays CUDA-graph capturable.

* Fixed the warp locomotion environment terminating one step earlier than the stable tasks and
  treating a torso below the negative termination height as a fall.
