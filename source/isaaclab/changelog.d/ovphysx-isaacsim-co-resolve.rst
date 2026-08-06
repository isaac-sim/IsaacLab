Changed
^^^^^^^

* Widened the ``packaging`` override to ``>=20,<27`` and removed the ``isaacsim`` /
  ``ovphysx`` conflict, so both extras now resolve into a single environment.
  ``uv run --extra isaacsim --extra ovphysx`` no longer forks the resolution, which
  lets one install run both Isaac Sim and OvPhysX tasks.
