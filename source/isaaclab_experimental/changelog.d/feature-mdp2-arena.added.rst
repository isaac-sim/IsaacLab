* Added cameras to :class:`~isaaclab_experimental.envs.MDP2Env`: their data types are scene fields that terms read,
  such as ``camera.rgb``, and the environment renders them once per control step at
  :attr:`~isaaclab_experimental.envs.Step.RENDER`, inside the step's CUDA graph.
  :attr:`~isaaclab_experimental.envs.ResetStep.RENDER` renders them again after resets.
* Added :class:`~isaaclab_experimental.mdp2.ObservationGroup`, whose ``concatenate=False`` keeps a buffer per term,
  with the term's own shape and element type, such as ``uint8`` images, observed as a dict of tensors, and
  :data:`~isaaclab_experimental.mdp2.mdp.image_rgb`.
* Added :class:`~isaaclab_experimental.mdp2.Action`, action terms whose hooks turn their columns of the action into
  writes, and :data:`~isaaclab_experimental.mdp2.mdp.binary_joint_position`, a binary gripper command with a
  threshold.
* Added :func:`~isaaclab_experimental.mdp2.compose`, which merges partial task configs, such as those of a scene, an
  embodiment, and a task, in order.
* Added the progress quantity :data:`~isaaclab_experimental.mdp2.mdp.sequence`, and
  :data:`~isaaclab_experimental.mdp2.mdp.joint_pos_normalized`.
* Added an episode budget to :class:`~isaaclab_experimental.envs.MDP2RLEnv` for evaluation, ``episode_budget``: worlds
  stop starting episodes once the budget is spent, and every episode is recorded on the device, with the quantities
  marked ``Quantity(record=True)``, in :meth:`~isaaclab_experimental.envs.MDP2RLEnv.episode_records`.
