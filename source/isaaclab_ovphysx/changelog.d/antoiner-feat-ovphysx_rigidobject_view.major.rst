Changed
^^^^^^^

* **Breaking:** :attr:`~isaaclab_ovphysx.assets.RigidObject.root_view` now returns an
  :class:`~isaaclab_ovphysx.sim.views.OvPhysxView` binding manager instead of a raw
  ``dict`` mapping ``TensorType`` to ``TensorBinding``. The view owns all tensor-binding
  creation, caching, reads, and writes for the rigid object. Replace ``root_view[tensor_type]``
  with ``root_view.try_binding_for(tensor_type)`` /
  :meth:`~isaaclab_ovphysx.sim.views.OvPhysxView.get_attribute`.
