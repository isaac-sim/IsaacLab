Changed
^^^^^^^

* Widened the ``packaging`` override to ``>=20,<27`` and removed the ``isaacsim`` /
  ``ovphysx`` conflict, so both extras now resolve into a single environment.
  ``uv run --extra isaacsim --extra ovphysx`` no longer forks the resolution, which
  lets one install run both Isaac Sim and OvPhysX tasks.

Fixed
^^^^^

* Fixed intermittent ``pxr`` import failures (``extension class wrapper ...
  Tf_PyEnumWrapper has not been created yet``) by overriding ``usd-exchange`` to
  aarch64 only. It and ``usd-core`` each vendor a complete ``pxr`` runtime built
  against a different USD version, and ``newton[importers]`` pulled usd-exchange onto
  x86_64 unmarked, so the two overlapped and the resulting runtime varied per build.
