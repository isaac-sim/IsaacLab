Added
^^^^^

* Added :class:`~isaaclab_tasks.core.dexsuite.config.kuka_allegro.agents.models.SpatialSoftmaxCNNModel`,
  an RSL-RL actor whose convolutional feature map is reduced to per-channel keypoint coordinates
  rather than flattened. The latent stays at two numbers per channel whatever the feature-map size,
  which keeps higher input resolutions affordable and leaves room for a higher learning rate.
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

* Changed the dexsuite camera presets to a spatial-softmax actor at a fixed learning rate. Camera
  actors are limited by how large an encoder update they tolerate, which ties the learning rate to
  the size of the latent the encoder hands to the MLP, so ``single_camera`` and ``duo_camera`` now
  reduce the convolutional feature map to per-channel keypoint coordinates
  (:class:`~isaaclab_tasks.core.dexsuite.config.kuka_allegro.agents.models.SpatialSoftmaxCNNModel`)
  and run with ``schedule="fixed"`` and ``learning_rate=1e-4``. The adaptive KL schedule oscillates
  across the stability threshold and does not train a camera actor at all. Measured at 4096
  environments, iterations to 50% success drop by 23% for a single RGB camera and 46% for duo
  depth, with unchanged final success. The state presets are unchanged.
* Changed the dexsuite camera observation normalization to fixed affine maps. RGB now maps to
  ``[-0.5, 0.5]`` and depth to the same span instead of subtracting a per-frame mean, which keeps
  pixel values stationary and preserves the absolute-distance anchor for depth.
* **Breaking:** Replaced ``position_command_error_tanh`` and ``orientation_command_error_tanh``
  with :class:`~isaaclab_tasks.core.dexsuite.mdp.position_command_progress` and
  :class:`~isaaclab_tasks.core.dexsuite.mdp.orientation_command_progress`, which pay a fixed amount
  per increment of ground gained on the episode-best error so that losing and regaining ground
  earns nothing. Replace the removed terms with the progress terms and set ``min_improvement``
  instead of ``std``.
