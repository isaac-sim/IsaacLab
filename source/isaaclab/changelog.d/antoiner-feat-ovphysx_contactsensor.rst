Fixed
^^^^^

* Fixed
  :meth:`~isaaclab.sensors.SensorBase._register_callbacks` matching the
  ``OvPhysxManager`` class name with the substring ``"physx"`` and
  trying to import :class:`isaaclab_physx.physics.IsaacEvents` —
  failing in kitless mode where ``omni.physics.tensors`` is not
  available.  The PhysX-only ``PRIM_DELETION`` callback is now gated
  on an exact ``physics_mgr_cls.__name__ == "PhysxManager"`` match.
