Changed
^^^^^^^

* Widened the ``packaging`` override to ``>=20,<27`` and removed the ``ovphysx``
  conflicts with ``isaacsim`` and ``teleop``, so those extras now resolve into a
  single environment.

Fixed
^^^^^

* Fixed intermittent ``pxr`` import failures by overriding ``usd-exchange`` to
  aarch64 only. It and ``usd-core`` each vendor a complete ``pxr`` runtime, and
  ``newton[importers]`` pulled usd-exchange onto x86_64 unmarked.
