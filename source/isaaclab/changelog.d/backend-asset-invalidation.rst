Added
^^^^^

* Added a ``skip_forward`` argument to the abstract root, body, and joint state writers of
  :class:`~isaaclab.assets.BaseArticulation`, :class:`~isaaclab.assets.BaseRigidObject`, and
  :class:`~isaaclab.assets.BaseRigidObjectCollection` to defer cached-buffer invalidation when
  several writes are batched before a single forward pass.
* Added :func:`~isaaclab.utils.buffers.reset_timestamps` to invalidate a list of timestamped
  buffers in one call, shared by the backend asset data classes' cache-reset helpers.

Changed
^^^^^^^

* **Breaking:** Added abstract ``_reset_pose`` and ``_reset_velocity`` cache-invalidation hooks to
  :class:`~isaaclab.assets.BaseArticulationData`, :class:`~isaaclab.assets.BaseRigidObjectData`,
  and :class:`~isaaclab.assets.BaseRigidObjectCollectionData`. Custom simulation-backend subclasses
  must now implement both methods to remain instantiable: ``_reset_pose`` invalidates the
  pose-derived cached buffers and ``_reset_velocity`` the velocity-derived ones (see the Newton,
  PhysX, and OV PhysX data classes for reference implementations).
