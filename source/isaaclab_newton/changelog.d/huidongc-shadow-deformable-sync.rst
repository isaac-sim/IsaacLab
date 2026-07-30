Added
^^^^^

* Added a shadow Newton visualization path for PhysX/OVPhysX deformables that syncs
  SceneData nodal positions into ``particle_q``, using dual sim/vis particle layouts
  and barycentric remapping when volume visual meshes differ from tet simulation
  topology.

Changed
^^^^^^^

* Changed :meth:`~isaaclab_newton.physics.NewtonManager.update_visualization_state`
  to always copy SceneData transforms and points into the bound shadow
  ``body_q`` / ``particle_q`` buffers instead of aliasing backend arrays, so
  ``get_state()`` consumers keep stable buffer identities across syncs.
