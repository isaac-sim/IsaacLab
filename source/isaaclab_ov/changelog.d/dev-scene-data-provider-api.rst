Changed
^^^^^^^

* **Breaking:** :class:`~isaaclab_ov.renderers.OVRTXRenderer` now reads the
  Newton ``Model`` and ``State`` it binds OVRTX attributes against from
  :meth:`~isaaclab_newton.physics.NewtonManager.get_model` /
  :meth:`~isaaclab_newton.physics.NewtonManager.get_state` instead of the
  removed ``BaseSceneDataProvider.get_newton_model()`` /
  ``get_newton_state()``.
