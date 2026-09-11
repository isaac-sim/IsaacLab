Added
^^^^^

* Added the ``IsaacContrib-Pick-And-Place-Apple-H2-Sharpa`` and
  ``IsaacContrib-Pick-And-Place-Apple-H2-Sharpa-Eval`` environments, an RLinf GR00T pick-and-place
  task on the Unitree H2 + Sharpa Wave embodiment.
* Added ``isaaclab_tasks.contrib.h2_sharpa``, holding the joint-order metadata, articulation presets
  and camera presets shared by the H2 + Sharpa tasks.

Changed
^^^^^^^

* Changed the default ``model_path`` in ``isaaclab_ppo_gr00t_assemble_trocar.yaml`` from an absolute
  container-specific path to ``.pretrained_checkpoints/rlinf/Assemble_Trocar``, the location that
  ``scripts/reinforcement_learning/rlinf/setup_rlinf.py`` downloads the checkpoint to. Pass
  ``--model_path`` to use a checkpoint stored elsewhere.
* Changed the default ``model_path`` in ``isaaclab_ppo_gr00t_pick_and_place_apple.yaml`` from an
  absolute container-specific path to ``.pretrained_checkpoints/rlinf/pnp_apple_n15_sft_ckpt``,
  matching the assemble-trocar task.
