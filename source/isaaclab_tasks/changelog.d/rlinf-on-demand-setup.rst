Added
^^^^^

* Added the ``IsaacContrib-Pick-And-Place-Apple-H2-Sharpa`` and
  ``IsaacContrib-Pick-And-Place-Apple-H2-Sharpa-Eval`` environments, an RLinf GR00T pick-and-place
  task on the Unitree H2 + Sharpa Wave embodiment.
* Added ``isaaclab_tasks.contrib.h2_sharpa``, holding the joint-order metadata, articulation presets
  and camera presets shared by the H2 + Sharpa tasks.
* Added the ``IsaacContrib-Pack-AGX-Orin-H2-Sharpa`` and ``IsaacContrib-Pack-AGX-Orin-H2-Sharpa-Eval``
  environments, an RLinf GR00T N1.7 task packing an AGX Orin into its protective box with the
  Unitree H2 + Sharpa Wave embodiment.
* Added GR00T N1.7 configurations ``isaaclab_ppo_gr00t_pick_and_place_apple_n17`` and
  ``isaaclab_ppo_gr00t_pack_agx_orin_n17``, and ``isaaclab_tasks.contrib.h2_sharpa.gr00t_n17``
  registering the H2 + Sharpa modality layout for N1.7 checkpoints.
* Changed the N1.7 configurations to plain PPO (``enable_sft_co_train: False``): RLinf has no SFT
  dataloader for ``gr00t_n1d7``, so co-training fails at actor start. Re-enable it only with an N1.7
  SFT dataloader registered in RLinf.
* Added ``ISAACLAB_H2_SHARPA_ASSET_ROOT`` and ``ISAACLAB_PICK_AND_PLACE_APPLE_ASSET_ROOT`` to point
  the H2 + Sharpa tasks at a local mirror of their scene and robot assets, which are hosted outside
  the Isaac asset server.

Changed
^^^^^^^

* Changed the default ``model_path`` in ``isaaclab_ppo_gr00t_assemble_trocar.yaml`` from an absolute
  container-specific path to ``.pretrained_checkpoints/rlinf/Assemble_Trocar``, the location that
  ``scripts/reinforcement_learning/rlinf/setup_rlinf.py`` downloads the checkpoint to. Pass
  ``--model_path`` to use a checkpoint stored elsewhere.
* Changed the default ``model_path`` in ``isaaclab_ppo_gr00t_pick_and_place_apple.yaml`` from an
  absolute container-specific path to ``.pretrained_checkpoints/rlinf/pnp_apple_n15_sft_ckpt``,
  matching the assemble-trocar task.
