Changelog
---------

1.0.7 (2025-03-19)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Moved the GR1T2 robot task to a separate directory to prevent import of pinocchio when not needed. This allows use of IsaacLab Mimic in windows.


1.0.6 (2025-03-10)
~~~~~~~~~~~~~~~~~~

Added
^^^^^^^

* Added :class:`FrankaCubeStackIKAbsMimicEnv` and support for the GR1T2 robot task (:class:`PickPlaceGR1T2MimicEnv`).


1.0.5 (2025-03-10)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Refactored dataset generation code into leaner modules to prepare for Jupyter notebook.

Added
^^^^^

* Added ``Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-Mimic-v0`` environment for blueprint vision stacking.


1.0.4 (2025-03-07)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated data generator to support environments with multiple end effectors.
* Updated data generator to support subtask constraints based on DexMimicGen.


1.0.3 (2025-03-06)
~~~~~~~~~~~~~~~~~~

Added
^^^^^^

* Added absolute pose mimic environment for Franka cube stacking task (:class:`FrankaCubeStackIKAbsMimicEnv`)


1.0.2 (2025-01-10)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

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
