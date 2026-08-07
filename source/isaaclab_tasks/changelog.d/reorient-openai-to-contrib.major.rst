Added
^^^^^

* Added the ``openai`` preset to ``Isaac-Reorient-Cube-Shadow`` and
  ``Isaac-Reorient-Cube-Shadow-Direct``, which narrows the actor to the 42 quantities a
  physical hand can measure and gives the critic the full 187-dimensional simulator state.
  Select it with ``presets=openai``; it configures the environment and the RSL-RL agent
  together.
* Added :meth:`~isaaclab_tasks.core.reorient.reorient_direct_env.ReorientDirectEnv._compute_time_out`,
  which subclasses override when the episode budget depends on goal progress rather than
  on elapsed time.

Changed
^^^^^^^

* **Breaking:** Removed the four OpenAI Shadow Hand task identifiers in favor of two
  contributed tasks. ``Isaac-Reorient-Cube-Shadow-OpenAI-FF`` and
  ``Isaac-Reorient-Cube-Shadow-OpenAI-LSTM`` become
  ``IsaacContrib-Reorient-Cube-Shadow-OpenAI``, and their ``-Direct`` counterparts become
  ``IsaacContrib-Reorient-Cube-Shadow-OpenAI-Direct``. The feed-forward and recurrent
  policies now share one environment and are selected with ``--agent``, so
  ``--task Isaac-Reorient-Cube-Shadow-OpenAI-LSTM-Direct`` becomes
  ``--task IsaacContrib-Reorient-Cube-Shadow-OpenAI-Direct --agent rsl_rl_lstm_cfg_entry_point``.
  These tasks reproduce a specific sim-to-real setup -- 20 Hz control, action and
  observation noise, and an episode budget spent per goal -- which does not generalize to
  other reorientation work. The observation architecture they use does, and stays in the
  core task as ``presets=openai``.
* **Breaking:** Replaced the Shadow Hand ``obs_type`` string with the
  :attr:`reduced_obs` and :attr:`asymmetric_obs` flags, which are independent: a task may
  narrow the actor, add a privileged critic, or do both. ``obs_type="openai"`` becomes
  ``presets=openai``; ``obs_type="full"`` is the default.
