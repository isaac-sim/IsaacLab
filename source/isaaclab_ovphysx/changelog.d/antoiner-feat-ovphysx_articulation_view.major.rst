Changed
^^^^^^^

* **Breaking:** :attr:`~isaaclab_ovphysx.assets.Articulation.root_view` now returns an
  :class:`~isaaclab_ovphysx.sim.views.OvPhysxView` binding manager instead of a raw
  ``dict`` mapping ``TensorType`` to ``TensorBinding``. The view owns all tensor-binding
  creation, caching, reads, and writes for the articulation. Address bindings by attribute
  name or ``TensorType`` member through
  :meth:`~isaaclab_ovphysx.sim.views.OvPhysxView.try_binding_for` /
  :meth:`~isaaclab_ovphysx.sim.views.OvPhysxView.get_attribute` rather than indexing the
  dict, e.g. replace ``root_view[tensor_type]`` with
  ``root_view.try_binding_for(tensor_type)`` and ``tensor_type in root_view`` with
  ``root_view.try_binding_for(tensor_type) is not None``.
