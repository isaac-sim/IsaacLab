* Changed the H2 tasks to refresh camera observations after resets with ``num_rerenders_on_reset = 2``.
* Changed AGX material overrides to use ``UsdFileCfg.visual_material`` on prototypes before cloning,
  retaining the authored textures and bindings without a task-specific spawner.
* Changed ``assemble_trocar`` to use the shared RLinf asset bundle and a relative default checkpoint
  path, ``.pretrained_checkpoints/rlinf/Assemble_Trocar``. Use ``--model_path`` for another checkpoint.
