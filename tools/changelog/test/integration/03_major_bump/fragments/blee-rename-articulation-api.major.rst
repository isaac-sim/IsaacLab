Removed
^^^^^^^

* Removed deprecated module ``example.old_api`` (use :mod:`~example.api` instead).

Changed
^^^^^^^

* **Breaking:** :meth:`~example.Foo.bar` now returns a tuple ``(value, error)`` instead of raising.
