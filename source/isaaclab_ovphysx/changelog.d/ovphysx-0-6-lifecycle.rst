Fixed
^^^^^

* Sealed OVStage population before attaching it to OVPhysX so articulation and
  joint data are available at the selected read ordinal.
* Added support for the OVPhysX 0.6 ``warmup()`` API while retaining the
  released 0.5 ``warmup_gpu()`` path.
