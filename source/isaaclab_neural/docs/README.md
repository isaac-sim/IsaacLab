# Isaac Lab Neural Robot Dynamics

This extension (`isaaclab_neural`) runs Isaac Lab tasks on neural robot dynamics models from the `newton-neural`
library. A model steps the robot of a task in place of a numerical solver, or inside one. The library is the optional
`newton-neural` extra of the Isaac Lab source checkout, and it is imported when a model is built or recorded.

The extension provides two parts.

- `isaaclab_neural.physics` holds what makes a trained model a physics preset of a task. `NeRDSolverCfg` with the
  `NewtonNeRDManager` runs a model in place of the solver (`physics=newton_nerd`), and `MJWarpNeRDSolverCfg` runs a
  model inside MuJoCo Warp (`physics=newton_mjwarp_nerd`).
- `NeRDRecipeCfg` and `NeRDRecipe` are the settings and the code by which a task says how its model is recorded and
  trained. A task registers its settings as `nerd_recipe_cfg_entry_point`.

The scripts `record.py`, `train.py` and `evaluate.py` under `scripts/neural` read the recipe of the task given with
`--task`. Presets and recipes ship for Franka reach, ANYmal C flat walking and Allegro cube reorientation. See the
guide "Training on neural robot dynamics" in the Isaac Lab documentation for the workflow.
