Fixed
^^^^^

* Added an early validation error when Agibot place tasks use Newton-backed visualizers, whose shadow-model importer
  does not support the reversed gripper joints in the robot USD. Use ``--visualizer kit`` with Isaac Sim PhysX, or
  ``--visualizer none`` for headless execution.
* Reduced the default Agibot place environment count from 4096 to one and enabled physics replication to prevent
  interactive tools from exhausting host memory. Use ``--num_envs`` to configure a larger batch when needed.
