Added
^^^^^

* Added the H2 + Sharpa tasks ``IsaacContrib-Pick-And-Place-Apple-H2-Sharpa`` and
  ``IsaacContrib-Pack-AGX-Orin-H2-Sharpa``, their ``-Eval`` registrations, and GR00T N1.7 PPO configurations.
* Added shared H2 joint ordering and calibrated wrist cameras in ``isaaclab_tasks.contrib.h2_sharpa``.
  Actions controlled the 58 policy joints directly; reset targets held the remaining body joints.
* Added task-owned phase tracking through stateful termination terms, with sparse rewards reading
  phase transitions independently of reward weights.
* Added ``ISAACLAB_RLINF_DEMO_ASSET_ROOT`` for a local mirror of the shared RLinf scene assets and
  documented their Lightwheel CC BY-NC 4.0 license.

Changed
^^^^^^^

* Changed the H2 tasks to refresh camera observations after resets with ``num_rerenders_on_reset = 2``.
* Changed AGX material overrides to use ``UsdFileCfg.visual_material`` on prototypes before cloning,
  retaining the authored textures and bindings without a task-specific spawner.
* Changed ``assemble_trocar`` to use the shared RLinf asset bundle and a relative default checkpoint
  path, ``.pretrained_checkpoints/rlinf/Assemble_Trocar``. Use ``--model_path`` for another checkpoint.
