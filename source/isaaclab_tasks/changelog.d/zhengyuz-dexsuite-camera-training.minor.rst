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

* Changed the dexsuite camera presets to a spatial-softmax actor at a fixed learning rate.
  ``single_camera`` and ``duo_camera`` now reduce the convolutional feature map to per-channel
  keypoint coordinates
  (:class:`~isaaclab_tasks.core.dexsuite.config.kuka_allegro.agents.models.SpatialSoftmaxCNNModel`)
  and run with ``schedule="fixed"`` and ``learning_rate=7e-5``; the adaptive KL schedule does not
  converge for a camera actor. State-policy optimizer settings were unchanged.
* Changed the dexsuite camera observation normalization to fixed maps that no longer depend on the
  frame's own statistics. RGB is rescaled affinely to ``[-0.5, 0.5]`` and depth is squashed with a
  ``tanh`` transform over the same span, replacing per-frame mean subtraction. Pixel values are now
  comparable across frames and depth keeps an absolute-distance reference.
* Changed the dexsuite tracking rewards used by the shipped tasks to
  :class:`~isaaclab_tasks.core.dexsuite.mdp.position_command_progress` and
  :class:`~isaaclab_tasks.core.dexsuite.mdp.orientation_command_progress`, which pay a fixed amount
  per increment of ground gained on the best error so far, so losing and regaining ground earns
  nothing.

Deprecated
^^^^^^^^^^

* Deprecated ``position_command_error_tanh`` and ``orientation_command_error_tanh`` in favor of
  :class:`~isaaclab_tasks.core.dexsuite.mdp.position_command_progress` and
  :class:`~isaaclab_tasks.core.dexsuite.mdp.orientation_command_progress`. Swap the term and set
  ``min_improvement`` instead of ``std``.

Fixed
^^^^^

* Fixed the dexsuite progress rewards crediting against a stale reference after the tracked command
  resampled mid-episode. The baseline is now re-seeded whenever the command changes.
