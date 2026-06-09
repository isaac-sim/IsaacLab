Fixed
^^^^^

* Re-enabled both CPU and GPU coverage in CI for OVPhysX tests by tagging
  :file:`test/assets/test_articulation.py`,
  :file:`test/assets/test_rigid_object.py`,
  :file:`test/assets/test_rigid_object_collection.py`,
  :file:`test/sensors/test_contact_sensor.py`, and
  :file:`test/sim/test_views_xform_prim_ovphysx.py` with the new
  ``device_split`` pytest marker, which causes the CI driver to invoke each
  file once per device in separate subprocesses. Works around the
  ``ovphysx<=0.3.7`` process-global device lock (gap G5).
