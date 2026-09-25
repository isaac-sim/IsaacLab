Fixed
^^^^^

* Fixed :meth:`~isaaclab_newton.assets.Articulation.write_fixed_tendon_properties_to_sim_index` not notifying
  the solver of the change, so written fixed tendon stiffness and damping never reached the MuJoCo model.
* Fixed :meth:`~isaaclab_newton.assets.Articulation.write_fixed_tendon_properties_to_sim_index` failing when
  ``env_ids`` is None.

Changed
^^^^^^^

* Changed the fixed tendon limit stiffness, rest length, and offset setters and data properties of
  :class:`~isaaclab_newton.assets.Articulation` to explain why the Newton backend does not support them in
  their :class:`NotImplementedError`.
