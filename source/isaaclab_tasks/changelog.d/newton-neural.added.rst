* Added the ``newton_nerd`` physics preset to ``Isaac-Reach-Franka`` and ``IsaacContrib-Velocity-Flat-AnymalC``, and
  the ``newton_mjwarp_nerd`` physics preset to ``Isaac-Reorient-Cube-Allegro``. They run the tasks on neural robot
  dynamics models through :mod:`isaaclab_neural` and log their RSL-RL policies to the experiments
  ``reach_franka_nerd``, ``anymal_c_flat_nerd`` and ``allegro_hand_mjwarp_nerd``.
* Added the recipes of these models. Their settings ``FrankaReachNeRDRecipeCfg``, ``AnymalCFlatNeRDRecipeCfg`` and
  ``AllegroHandNeRDRecipeCfg`` are registered as ``nerd_recipe_cfg_entry_point`` for the scripts under
  ``scripts/neural``.
