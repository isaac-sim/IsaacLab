Added
^^^^^

* Added :func:`~isaaclab.test.launch.launch_kit` so test modules can share one Kit app per
  pytest process instead of each launching their own. It is idempotent: the first module to
  call it boots Kit and later modules receive the running app.
* Added the ``kit``, ``kit_cameras``, ``kitless``, and ``kit_solo`` pytest markers so a test
  file can declare its Kit launch configuration, plus a test that checks each file's markers
  against what it actually does at module scope.

Fixed
^^^^^

* Fixed ``test_operational_space.py`` assigning ``pytestmark`` twice, which silently dropped
  its ``arm_ci`` marker and kept the file out of the ARM CI lane.
