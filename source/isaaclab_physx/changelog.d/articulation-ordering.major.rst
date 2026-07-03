Added
^^^^^

* Added backend joint/body ordering introspection properties to
  :class:`~isaaclab_physx.assets.Articulation`.

Removed
^^^^^^^

* Removed the ``write_joint_state_data`` and ``write_joint_vel_data`` kernels
  from ``isaaclab_physx.assets.articulation.kernels``. Prefer the public-order
  asset write APIs (:meth:`~isaaclab.assets.Articulation.write_joint_position_to_sim_index`
  and its siblings), which apply the ordering conversion internally. Code that
  works directly with raw solver views can instead launch the public elementwise
  reorder kernels (the ``reorder_2d_user_to_backend`` /
  ``reorder_2d_backend_to_user`` and ``reorder_3d_user_to_backend`` /
  ``reorder_3d_backend_to_user`` family) from
  ``isaaclab.assets.articulation.ordering_kernels`` together with the asset's
  ordering maps.

Changed
^^^^^^^

* Changed :attr:`~isaaclab.assets.ArticulationData.body_mass` and
  :attr:`~isaaclab.assets.ArticulationData.body_inertia` to refresh from the
  simulation lazily, at most once per simulation step, instead of on every
  read. Values written directly through the tensor view become visible at the
  first read after the next simulation update; use the asset's
  :meth:`~isaaclab.assets.Articulation.set_masses_index` and
  :meth:`~isaaclab.assets.Articulation.set_inertias_index` for immediately
  coherent writes.
