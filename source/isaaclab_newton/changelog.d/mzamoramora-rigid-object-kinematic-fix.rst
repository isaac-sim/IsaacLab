Fixed
^^^^^

* Fixed :class:`~isaaclab_newton.assets.RigidObjectData` crashing with
  ``IndexError: tuple index out of range`` for kinematic-enabled
  single-body fixed-base rigid objects. The ``is_fixed_base`` branch
  in ``_create_simulation_bindings`` indexed ``[:, 0, 0]`` assuming a
  3D ``(count, links, 1)`` layout, but Newton returns a 2D
  ``(count, links)`` array when the view contains a single body.
  Dispatch on actual ``ndim`` instead so both multi-link fixed-base
  articulations and single-body kinematic rigid objects are handled
  correctly. Also fixes the matching no-velocity fallback in
  ``_create_buffers``: ``_sim_bind_body_com_vel_w`` is now allocated
  as ``(num_instances, 1)`` to match the
  ``derive_body_acceleration_from_body_com_velocities`` kernel's
  2D signature.
