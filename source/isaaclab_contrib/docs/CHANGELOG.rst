Changelog
---------

0.3.2 (2026-05-12)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Removed ``_patched_reset`` monkey-patch in RLinf extension; use
  ``num_rerenders_on_reset`` env config instead.


0.3.1 (2026-05-09)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated TacSL visuotactile sensor camera configuration and examples to use
  :class:`~isaaclab.sensors.CameraCfg` and :class:`~isaaclab.sensors.Camera`
  instead of deprecated tiled-camera aliases.


0.3.0 (2026-02-13)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated multirotor asset and TacSL visuotactile sensor to wrap warp data
  property accesses with ``wp.to_torch()``.


0.2.1 (2026-02-03)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated the multirotor asset to use the new base classes from the isaaclab_physx package.


0.2.0 (2026-01-30)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated the multirotor asset to use the new base classes from the isaaclab_physx package.


0.1.0 (2026-01-30)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^


* Changed the quaternion ordering to match warp, PhysX, and Newton native XYZW quaternion ordering.


0.0.2 (2026-01-28)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :mod:`isaaclab_contrib.sensors.tacsl_sensor` module with the TacSL tactile sensor implementation
  from :cite:t:`si2022taxim`.


0.0.1 (2025-12-17)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added initial implementation for multi rotor systems.
