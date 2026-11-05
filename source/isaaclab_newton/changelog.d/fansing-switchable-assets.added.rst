* Added Newton support for :class:`~isaaclab.sim.SwitchableAssetsCfg`, also beneath another asset: the object is one
  body whose shape variants are its candidates, selected with
  :meth:`~isaaclab_newton.assets.RigidObject.replace_mesh_to_sim` or
  :meth:`~isaaclab_newton.physics.NewtonManager.select_body_variants`. Renderers receive the candidate prims with the
  unselected ones out of view, and Newton viewers follow the selection through
  :meth:`~isaaclab_newton.physics.NewtonBackend.notify_viewer`. The candidates of an object must all be kinematic or
  all dynamic.
