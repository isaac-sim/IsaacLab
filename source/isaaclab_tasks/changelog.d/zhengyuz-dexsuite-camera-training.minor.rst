Added
^^^^^

* Added :class:`~isaaclab_tasks.core.dexsuite.mdp.SuccessMonitorCfg` and its
  :class:`~isaaclab_tasks.core.dexsuite.mdp.SuccessMonitor` implementation, which draws banked
  reset states by measured success rate so episodes restart from the states the policy solves
  about half the time.
* Added :class:`~isaaclab_tasks.core.dexsuite.mdp.GraspTravelDistanceCfg` and
  :func:`~isaaclab_tasks.core.dexsuite.mdp.grasp_travel_distance`, a reset-bank diversity feature
  that spreads banked states over hand-to-object and object-to-goal distance.
* Added ``disable_observation_noise_terms`` to the dexsuite ADR curriculum config, which drops the
  noise-scheduling terms whose addresses no longer resolve when observation corruption is off.

Changed
^^^^^^^

* Changed the dexsuite camera presets to a shared encoder and a fixed learning rate. Camera actor
  stability depends on bounding the encoder update magnitude, so ``single_camera`` and
  ``duo_camera`` now use a ``[16, 32, 32]`` convolutional encoder with ``schedule="fixed"`` and
  ``learning_rate=5e-5``; the adaptive KL schedule oscillates across the stability threshold and
  does not train a camera actor. The state presets are unchanged.
* Changed the dexsuite camera observation normalization to fixed affine maps. RGB now maps to
  ``[-0.5, 0.5]`` and depth to the same span instead of subtracting a per-frame mean, which keeps
  pixel values stationary and preserves the absolute-distance anchor for depth.
* **Breaking:** Replaced ``position_command_error_tanh`` and ``orientation_command_error_tanh``
  with :class:`~isaaclab_tasks.core.dexsuite.mdp.position_command_progress` and
  :class:`~isaaclab_tasks.core.dexsuite.mdp.orientation_command_progress`, which pay a fixed amount
  per increment of ground gained on the episode-best error so that losing and regaining ground
  earns nothing. Replace the removed terms with the progress terms and set ``min_improvement``
  instead of ``std``.
