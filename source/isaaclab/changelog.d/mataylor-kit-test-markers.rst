Added
^^^^^

* Added the ``kit``, ``kit_cameras``, and ``solo`` pytest markers, and the
  :mod:`isaaclab.test.kit` plugin that acts on them. A test file that needs Isaac Sim now
  declares it in its module-level ``pytestmark``; the plugin reads that declaration out of the
  file's source and boots Kit before pytest imports the module, so files sharing a launch
  configuration share one app instead of each starting their own. Test files no longer
  construct :class:`~isaaclab.app.AppLauncher` at module scope, and tests that need the app
  object request the new ``kit_app`` fixture.
* Added ``test_kit_marker_contract.py`` and ``test_kit_plugin.py``, which keep a file's markers
  from drifting from what it does at module scope, and check that the app is started before the
  module that needs it is imported.

Changed
^^^^^^^

* Changed ``isaaclab --test`` to drive the new ``tools/run_tests.py``. With no arguments it
  runs the whole suite; ``--job <name>`` runs a single CI lane locally, ``--list-jobs`` lists
  them, and directories run an ad-hoc selection. It previously ran ``pytest tools``, which went
  through the CI orchestrator and silently dropped any pytest arguments passed after it.
* Changed the runner to group same-marker test files into a single pytest invocation rather
  than giving every file its own process, so Kit startup is paid once per group. Only files
  carrying the new markers are grouped; every other file keeps a process of its own, and a file
  a dead group never reached is re-run individually. Set ``ISAACLAB_TEST_BATCH_KIT=0`` to turn
  the grouping off.

Fixed
^^^^^

* Fixed ``test_operational_space.py`` assigning ``pytestmark`` twice, which silently dropped
  its ``arm_ci`` marker and kept the file out of the ARM CI lane.
