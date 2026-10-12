* **Breaking:** Removed ``NewtonManager.unregister_post_step_callback``; pass the callback returned by a
  ``register_*_callback`` method to :meth:`~isaaclab_newton.physics.NewtonManager.unregister_step_callback`.
* **Breaking:** Removed ``NewtonManager.add_frame_transform_sensor``, ``instantiate_builder_from_stage``,
  ``get_num_envs``, ``get_dt``, ``get_solver_dt``, ``is_fabric_enabled``, ``provides_implicit_damping`` and
  ``add_world_builder_hook``. Use :func:`~isaaclab_newton.cloner.newton_builder_world_hook` to extend worlds.
