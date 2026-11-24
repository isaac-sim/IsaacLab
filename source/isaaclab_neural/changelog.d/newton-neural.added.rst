* Added the :mod:`isaaclab_neural` package, which runs Newton tasks on neural robot dynamics models of the
  ``newton-neural`` library. The library is the optional ``newton-neural`` extra and is imported when a model is
  built or recorded, so tasks that declare a model preset import without it.
* Added :class:`~isaaclab_neural.physics.NeRDSolverCfg` and :class:`~isaaclab_neural.physics.NewtonNeRDManager` for
  a physics preset in which a model takes the place of the solver. ``class_type`` of a
  :class:`~isaaclab_newton.physics.NewtonCfg` selects the manager, which steps the model once per environment step,
  so a task keeps its ``sim.dt`` and ``decimation``.
* Added :class:`~isaaclab_neural.physics.MJWarpNeRDSolverCfg` for a physics preset in which a model of the free
  motion of a robot runs inside MuJoCo Warp, which still solves the contacts.
* Added :class:`~isaaclab_neural.NeRDRecipeCfg` and :class:`~isaaclab_neural.NeRDRecipe`, the settings and the code
  by which a task says how its model is recorded and trained, and :class:`~isaaclab_neural.RecordingSimulation`,
  the recording environment as a recipe sees it. A task registers its settings as ``nerd_recipe_cfg_entry_point``.
* Added the scripts ``record.py``, ``train.py`` and ``evaluate.py`` under ``scripts/neural``, which record the data
  of the model of a task, train the model, and evaluate the policies that learned on it. Each takes ``--task``.
