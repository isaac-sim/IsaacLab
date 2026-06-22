Added
^^^^^

* Added :class:`~isaaclab_ovphysx.sim.views.OvPhysxView`, a string-keyed view over the
  OVPhysX tensor bindings. Attributes are addressed by the lowercased ``TensorType``
  name (e.g. ``view.get_attribute("articulation_dof_stiffness")`` /
  ``view.set_attribute("rigid_body_pose", values, mask=...)``), bringing the OVPhysX
  binding surface closer to the Newton selection API. Prototype tracked by the design
  note at ``docs/superpowers/specs/2026-06-17-ovphysx-view-design.md``.
