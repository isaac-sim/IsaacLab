Added
^^^^^

* Added ``tools/task_discovery.py``, which enumerates registered training tasks and the
  backend combinations they support. ``discover_tasks(resolve=False)`` reports what each
  task declares; ``resolve=True`` additionally builds each combination and runs the
  runtime validator, so combinations that are declared but cannot run are excluded.
  Automatic selectors such as ``physics=physx`` and ``renderer=rtx`` are reported
  separately from concrete backends, letting callers exclude aliases without hardcoding
  their names.
