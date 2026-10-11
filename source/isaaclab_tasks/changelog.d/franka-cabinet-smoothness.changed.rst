* **Breaking:** Updated both Franka drawer workflows with the existing EMA joint-position action,
  stronger motion and joint-limit penalties, increased arm damping, and 16-second episodes.
  Actions now map to soft joint limits and are smoothed before application. Retrain existing
  policies for the changed action semantics, rewards, and reset pose. Direct configurations
  now use ``arm_action`` in place of ``arm_joint_names`` and ``arm_action_scale``.
  The 0.3 rad/s actuator limit is not a hard physical speed bound in MJWarp.
* Gave both cabinet workflows light-brown satin fronts, a cream body, and bronze hardware
  using shared materials, without changing cabinet physics or the default floor.
