Changed
^^^^^^^

* Inherits the base
  :attr:`~isaaclab.assets.BaseArticulationData.body_link_jacobian_w`,
  :attr:`~isaaclab.assets.BaseArticulationData.body_com_jacobian_w`,
  :attr:`~isaaclab.assets.BaseArticulationData.mass_matrix`, and
  :attr:`~isaaclab.assets.BaseArticulationData.gravity_compensation_forces`
  :class:`NotImplementedError` defaults — ovphysx's OmniGraph-based view
  does not expose articulation Jacobians, mass matrices, or gravity
  compensation. Use the PhysX or Newton backends for task-space
  controllers.
