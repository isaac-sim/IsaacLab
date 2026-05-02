Changelog
---------

2.0.0 (2026-04-30)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~example.AnotherSensor` for proximity detection.

Changed
^^^^^^^

* **Breaking:** :meth:`~example.Foo.bar` now returns a tuple ``(value, error)`` instead of raising.

Removed
^^^^^^^

* Removed deprecated module ``example.old_api`` (use :mod:`~example.api` instead).

Fixed
^^^^^

* Fixed a deadlock in :class:`~example.Pool` under high concurrency.


1.2.3 (2026-01-15)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~example.OldThing` for an earlier feature.
