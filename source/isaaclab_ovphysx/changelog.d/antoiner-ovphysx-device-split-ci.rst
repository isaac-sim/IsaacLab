Fixed
^^^^^

* Re-enabled both CPU and GPU coverage in CI for
  :file:`test/sim/test_views_xform_prim_ovphysx.py` and
  :file:`test/sensors/test_contact_sensor.py` by tagging them with the new
  ``device_split`` pytest marker, which causes the CI driver to invoke each
  file once per device in separate subprocesses. Works around the
  ``ovphysx<=0.3.7`` process-global device lock (gap G5).
