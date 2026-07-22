Added
^^^^^

* Added :class:`~isaaclab.physics.PhysxAutoCfg` for launch-time selection
  between Isaac Sim PhysX and OvPhysX.

Changed
^^^^^^^

* Changed the ``physx`` launcher selector to resolve to Isaac Sim PhysX when a
  Kit renderer or Kit viewer is requested and to OvPhysX otherwise. Use
  ``isaacsim_physx`` to force Isaac Sim PhysX.
