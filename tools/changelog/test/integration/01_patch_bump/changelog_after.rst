Changelog
---------

1.2.4 (2026-04-30)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Tightened error message in :class:`~example.Foo` when a required argument is missing.

Fixed
^^^^^

* Fixed missing GPU sync in :func:`~example.refresh_buffers` that occasionally returned stale data.
* Fixed off-by-one in :meth:`~example.Foo.bar` when the input list was empty.


1.2.3 (2026-01-15)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~example.OldThing` for an earlier feature.
