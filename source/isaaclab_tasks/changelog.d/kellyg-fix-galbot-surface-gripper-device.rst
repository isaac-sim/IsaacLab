Fixed
^^^^^

* Fixed surface-gripper stack tasks to select CPU simulation by default and reject unsupported GPU overrides before
  simulator initialization. Task-defined simulation devices are now preserved by :func:`parse_env_cfg` when no
  explicit device override is provided.
