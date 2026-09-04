Added
^^^^^

* Added :attr:`~isaaclab_ov.tensor_types.DOF_DRIVE_TYPE`,
  :attr:`~isaaclab_ov.tensor_types.DOF_DRIVE_MODEL`,
  :attr:`~isaaclab_ov.tensor_types.BODY_DISABLE_GRAVITY`,
  :attr:`~isaaclab_ov.tensor_types.CONTACT_OFFSET`,
  :attr:`~isaaclab_ov.tensor_types.REST_OFFSET`,
  :attr:`~isaaclab_ov.tensor_types.RIGID_BODY_DISABLE_GRAVITY`,
  :attr:`~isaaclab_ov.tensor_types.RIGID_BODY_CONTACT_OFFSET`, and
  :attr:`~isaaclab_ov.tensor_types.RIGID_BODY_REST_OFFSET` tensor type aliases,
  documenting the shape, dtype and units of each.

Fixed
^^^^^

* Fixed :class:`~isaaclab_ov.sim.views.OvPhysxView` routing eight CPU-resident
  tensor types to the simulation device. The per-collision-shape contact and rest
  offsets, the articulation and rigid-body gravity-disable flags, and the DOF drive
  type and drive model are CPU-resident even on a GPU simulation, but were absent
  from the internal CPU-only classification. Reads and writes of these types
  incurred a hidden per-call host-to-device staging copy, and a correctly placed
  host buffer was rejected with ``OvPhysxView.DeviceMismatch``. Residency was
  measured on a GPU simulation by counting CUDA memcpys around a binding read.
* **Breaking:** Fixed ``articulation_dof_drive_type`` not being classified as
  read-only. The underlying tensor type is read-only, but
  :meth:`~isaaclab_ov.sim.views.OvPhysxView.set_attribute` previously accepted
  writes to it and silently forwarded them. Such calls now raise
  ``OvPhysxView.ReadOnlyAttribute``. Remove any write to this attribute; drive
  type is authored through the USD drive schema, not the tensor path.
