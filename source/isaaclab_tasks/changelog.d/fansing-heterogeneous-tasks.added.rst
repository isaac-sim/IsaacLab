* Added the ``contrib/heterogeneous_tasks`` tasks for vision-language-action policies on Newton: 256 example
  pick-and-place tasks on furniture (tables, desks, cabinets, shelves) generated from the RoboLab and Omniverse
  object libraries, some of which require the placed objects upright, registered as
  ``IsaacContrib-VLA-PickPlace<NNN>-Franka``, and ``IsaacContrib-VLA-Heterogeneous-Franka``, whose environments each run
  one of them with the success of all tasks evaluated together. The tasks load their assets from the HeteroTask
  dataset on Hugging Face, from its local Hugging Face cache when downloaded, or from the location in the
  ``VLA_BENCHMARKS_ASSETS_DIR`` environment variable, and use
  :obj:`~isaaclab_assets.robots.franka.FRANKA_ROBOTIQ_GRIPPER_CFG`.
* Defined all 256 benchmark tasks as inline records over one shared flat asset catalog and environment
  configuration. Removed per-task modules and environment subclasses while preserving individual Gym IDs,
  instructions, poses, episode durations and success terms through record selectors.
