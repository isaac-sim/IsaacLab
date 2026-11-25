* Added the :mod:`isaaclab_neural` package, which runs Newton tasks on neural robot dynamics models of the
  ``newton-neural`` library. The library is the optional ``newton-neural`` extra and is imported when a model is
  built, recorded or fitted, so tasks that declare a model preset import without it.
* Added :class:`~isaaclab_neural.physics.NeRDSolverCfg` for a physics preset in which a model takes the place of
  the solver. It runs on the ordinary Newton manager, and the preset declares one model frame as ``sim.dt`` with
  ``decimation=1`` and ``num_substeps=1``.
* Added :class:`~isaaclab_neural.physics.MJWarpNeRDSolverCfg` for a physics preset in which a model of the free
  motion of a robot runs inside MuJoCo Warp, which still solves the contacts.
* Added :class:`~isaaclab_neural.NeRDCfg` for the model definition and :class:`~isaaclab_neural.NeRDTrainingCfg`
  for its optimization, which a task registers as ``nerd_cfg_entry_point`` and ``nerd_training_cfg_entry_point``.
* Added exploration action, reset, termination and recorder terms in ``isaaclab_neural.record`` for the recording
  environment that a task registers as ``nerd_recording_cfg_entry_point``. The recorder term writes batched native
  trajectory shards and describes the recorded robot in the dataset.
* Added the scripts ``record.py`` and ``train.py`` under ``scripts/neural``, which record the data of the model of
  a task and fit the model definition to it. Each takes ``--task``.
