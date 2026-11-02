* Added the ``contrib/heterogeneous_tasks`` tasks for vision-language-action policies on Newton: 256 example
  pick-and-place tasks on furniture (tables, desks, cabinets, shelves) generated from the RoboLab and Omniverse
  object libraries, some of which require the placed objects upright, registered as
  ``IsaacContrib-VLA-PickPlace<NNN>-Franka``, and ``IsaacContrib-VLA-Heterogeneous-Franka``, whose environments each run
  one of them with the success of all tasks evaluated together. The tasks load their assets from the HeteroTask
  dataset on Hugging Face, or from the location in the ``VLA_BENCHMARKS_ASSETS_DIR`` environment variable.
