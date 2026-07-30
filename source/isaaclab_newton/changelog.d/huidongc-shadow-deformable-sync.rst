Added
^^^^^

* Added a shadow Newton visualization path for PhysX/OVPhysX deformables that syncs
  SceneData nodal positions into ``particle_q``, using dual sim/vis particle layouts
  and barycentric remapping when volume visual meshes differ from tet simulation
  topology.

Changed
^^^^^^^

* Changed :meth:`~isaaclab_newton.physics.NewtonManager.update_visualization_state`
  to sync transforms and points with ``allow_passthrough=False`` so shadow
  ``body_q`` / ``particle_q`` buffers stay bound for ``get_state()`` consumers.
  Callers that relied on passthrough aliasing must pass ``allow_passthrough=True``
  explicitly.
