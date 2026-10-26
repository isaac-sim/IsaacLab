* Fixed :meth:`~isaaclab_newton.assets.Articulation.reset` with only ``env_mask`` resetting the actuator state
  (delay buffers, network history, and Newton-native actuator state) of every environment.
* Fixed :meth:`~isaaclab_newton.assets.RigidObject.reset` and
  :meth:`~isaaclab_newton.assets.RigidObjectCollection.reset` ignoring ``env_mask`` and clearing the external
  wrenches of every environment.
* Fixed fixed-tendon position targets commanded inside a captured CUDA graph reaching MuJoCo's controls only at
  capture time. :meth:`~isaaclab_newton.assets.Articulation.write_data_to_sim` now writes the buffered targets on
  every step when the articulation has tendon actuators, instead of behind a host-side flag.
