* Fixed the fixed-tendon mask setters of :class:`~isaaclab_newton.assets.Articulation` accepting data whose shape
  does not match the articulation.
* Fixed model-property writes recorded into a graph capture refreshing the solver only on the first replay:
  :meth:`~isaaclab_newton.physics.NewtonManager.add_model_change` now records the solver refresh of that change, so
  every replay applies it.
