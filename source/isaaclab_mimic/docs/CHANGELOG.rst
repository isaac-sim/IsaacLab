Changelog
---------

1.0.12 (2025-07-31)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``from __future__ import annotations`` to utils.py to fix Sphinx
  doc warnings for IsaacLab Mimic docs.


1.0.11 (2025-07-17)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated test_selection_strategy.py and test_generate_dataset.py test cases to pytest format.
* Updated annotate_demos.py script to return the number of successful task completions as the exit code to support check in test_generate_dataset.py test case.


1.0.10 (2025-07-08)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated generate dataset script to cancel remaining async tasks before closing the simulation app.


1.0.9 (2025-05-20)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Mimic-v0`` environment for Cosmos vision stacking.


1.0.8 (2025-05-01)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`NutPourGR1T2MimicEnv` and :class:`ExhaustPipeGR1T2MimicEnv` for the GR1T2 nut pouring and exhaust pipe tasks.
* Updated instruction display to support all XR handtracking devices.


1.0.7 (2025-03-19)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Moved the GR1T2 robot task to a separate directory to prevent import of pinocchio when not needed. This allows use of IsaacLab Mimic in windows.


1.0.6 (2025-03-10)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

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
