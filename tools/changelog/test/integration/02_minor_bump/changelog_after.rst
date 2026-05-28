Changelog
---------

1.3.0 (2026-04-30)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~example.NewSensor` for IMU-based proprioception.
* Added :func:`~example.helper` utility for batched coordinate transforms.

Changed
^^^^^^^

* Documented thread-safety guarantees for :class:`~example.Worker`.

Fixed
^^^^^

* Fixed a NaN propagation in :meth:`~example.Sensor.update`.


1.2.3 (2026-01-15)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~example.OldThing` for an earlier feature.
