* Added ``--cosmos`` and related options to ``isaaclab train`` and ``isaaclab play`` to generate a camera task's
  policy images with a running Cosmos service, without changes to the task. ``--num_envs`` environments are
  generated as one batch by a service started with ``--max-views``; ``--cosmos_control`` takes ``depth``,
  ``edge``, or ``blur``, and ``--cosmos_transport`` ``auto``, ``cuda_ipc``, or ``socket``.
