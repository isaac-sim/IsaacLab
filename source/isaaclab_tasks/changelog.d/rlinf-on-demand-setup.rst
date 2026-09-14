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
* Added ``POLICY_STATE_TO_ACTION_INDICES`` to ``isaaclab_tasks.contrib.h2_sharpa.gr00t_n17``, publishing where the policy's 58-D joint state lands in H2's 75-D action vector. The RLinf extension reads it to build a hold-the-current-pose action; H2 interleaves the two hands, so the state is not a contiguous slice of the action.
* Added ``isaaclab_tasks.contrib.rlinf_assets``, the asset roots the three RLinf post-training tasks
  share, and ``ISAACLAB_RLINF_DEMO_ASSET_ROOT`` to point them at a local mirror.

Changed
^^^^^^^

* Changed ``assemble_trocar`` to draw its scene from the same asset bundle as the H2 + Sharpa tasks
  instead of its own pinned Isaac Healthcare release, so props the tasks have in common are fetched
  and cached once. The files are byte-identical to the ones it used before.
* Changed ``isaaclab_ppo_gr00t_pack_agx_orin_n17`` to evaluate over 448-step episodes, matching the evaluation the task's reference checkpoints were scored with. ``env.train`` keeps its 416-step rollout, which ``global_batch_size`` is sized for.
* Changed the default ``model_path`` in ``isaaclab_ppo_gr00t_assemble_trocar.yaml`` from an absolute
  container-specific path to ``.pretrained_checkpoints/rlinf/Assemble_Trocar``, the location that
  ``scripts/reinforcement_learning/rlinf/setup_rlinf.py`` downloads the checkpoint to. Pass
  ``--model_path`` to use a checkpoint stored elsewhere.
* Changed the H2 + Sharpa robot USD to carry its own material bindings. The wrist camera brackets and
  spacers are black plastic on the real robot but shipped without a material, so they fell back to the
  renderer's default white surface; twelve further meshes had no material either. The bindings are now
  authored in the asset, which removes ``align_robot_arm_material`` from the AGX Orin packing task and
  gives every task the same robot appearance. Re-download the asset bundle to pick it up.
* Changed the NuRec backdrop capture to carry its dimmed colour-correction matrix. The AGX Orin
  packing task scaled it at startup so the back panel sat below the tabletop the way the real
  recording shows; the scale is now baked into the capture, which removes
  ``align_backdrop_radiance`` and gives the pick-and-place apple task the same backdrop.
