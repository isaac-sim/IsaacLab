Changed
^^^^^^^

* **Breaking:** :attr:`~isaaclab_ovphysx.assets.RigidObjectCollection.root_view` now returns an
  :class:`~isaaclab_ovphysx.sim.views.OvPhysxView` binding manager instead of a raw ``dict``
  mapping ``TensorType`` to ``TensorBinding``. The view wraps the fused multi-prim bindings
  (``prim_paths=[...]``) and stores each under the collection's ``LINK_*``/``BODY_*`` data-class
  key via ``key_aliases`` (mapped from the underlying ``RIGID_BODY_*`` type). Replace
  ``root_view[tensor_type]`` with ``root_view.try_binding_for(tensor_type)``.
