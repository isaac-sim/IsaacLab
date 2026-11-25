* Added the ``newton_nerd`` physics preset to ``Isaac-Reach-Franka`` and ``IsaacContrib-Velocity-Flat-AnymalC``, and
  the ``newton_mjwarp_nerd`` physics preset to ``Isaac-Reorient-Cube-Allegro``. They run the tasks on neural robot
  dynamics models through :mod:`isaaclab_neural` and log their RSL-RL policies to the experiments
  ``reach_franka_nerd``, ``anymal_c_flat_nerd`` and ``allegro_hand_mjwarp_nerd``.
* Added the model definitions, fitting settings and recording environments of these models in the ``dynamics`` and
  ``record`` folders of the three tasks, registered as ``nerd_cfg_entry_point``, ``nerd_training_cfg_entry_point``
  and ``nerd_recording_cfg_entry_point`` for the scripts under ``scripts/neural``.
* Added the ``env_cfg_entry_point`` argument to :func:`~isaaclab_tasks.utils.hydra.resolve_task_config` and
  :func:`~isaaclab_tasks.utils.hydra.register_task`, which selects the registered environment configuration that
  a workflow composes.
