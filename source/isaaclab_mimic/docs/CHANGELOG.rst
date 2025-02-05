Changelog
---------

1.0.3 (2025-02-05)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^^^

* Fixed the issue where environment reset invoked in mimic's asyncio data generation task throws error when
  camera rendering is enabled.


1.0.2 (2025-01-10)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^^^

* Fixed test_selection_strategy.py test case by starting omniverse app to import needed dependencies.


1.0.1 (2024-12-16)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Removed the custom :meth:`get_object_poses` function in the:class:`FrankaCubeStackIKRelMimicEnv`
  class to use the default implementation from the :class:`ManagerBasedRLMimicEnv` class.


1.0.0 (2024-12-06)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Add initial version of Isaac Lab Mimic
