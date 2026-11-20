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
* Added tasks of tasks to MDP2: a :class:`~isaaclab_experimental.mdp2.TaskCfg` may say when it is ``done`` (with
  ``hold``, ``score``, and ``final``) and hold named subtasks, which run together
  (:func:`~isaaclab_experimental.mdp2.compose`, with ``require``), one after another
  (:func:`~isaaclab_experimental.mdp2.sequence`), or in their own worlds (:func:`~isaaclab_experimental.mdp2.split`).
  :class:`~isaaclab_experimental.mdp2.Milestone` is a task with only a completion. The uses of a task of a sequence run
  while it does, and restart when it starts. Terms read the progress of every task, such as
  ``mdp2.progress.grasp.done``, and :class:`~isaaclab_experimental.envs.MDP2RLEnv` logs it as ``Progress/<task>`` and
  records the step at which each task was done.
* Added :meth:`~isaaclab_experimental.mdp2.MDP.spec` and :meth:`~isaaclab_experimental.mdp2.MDP.observation_mask`,
  which name the columns of the observation buffers and the columns each world writes.
* Added :data:`~isaaclab_experimental.mdp2.mdp.joint_pos_normalized`, :data:`~isaaclab_experimental.mdp2.mdp.is_true`,
  and :data:`~isaaclab_experimental.mdp2.mdp.one_hot`.
* Added an episode budget to :class:`~isaaclab_experimental.envs.MDP2RLEnv` for evaluation, ``episode_budget``: worlds
  stop starting episodes once the budget is spent, and every episode is recorded on the device, with the quantities
  marked ``Quantity(record=True)``, in :meth:`~isaaclab_experimental.envs.MDP2RLEnv.episode_records`.
* Added OptiX cameras to :class:`~isaaclab_experimental.envs.MDP2Env`: cameras with
  :class:`~isaaclab_newton.renderers.NewtonOptixRendererCfg` render between the CUDA graphs of a step, since OptiX
  refits its acceleration structures on the host.
* Added runtime values to MDP2: a param or reward weight declared :class:`~isaaclab_experimental.mdp2.Tunable` holds
  one value per world, which :func:`~isaaclab_experimental.mdp2.tune` and
  :meth:`~isaaclab_experimental.envs.MDP2Env.tune` change at each world's next reset or from the next step, while
  captured steps keep replaying. Terms read it as a field, :func:`~isaaclab_experimental.mdp2.tunable`, and each world
  counts the tunings it applied, recorded as ``revision``.
* Added ``when`` to MDP2 term uses: a reward, termination, observation, command, or interval event runs only in the
  worlds where a field, such as a mode command, was positive at the start of the step, and restarts its state, timer,
  command, and history where it turns on. ``when=mdp2.Tunable(True)`` is a switch per world that
  :func:`~isaaclab_experimental.mdp2.tune` turns on and off.
* Added ``repeat`` to MDP2 tasks: a task that repeats counts its completions as ``mdp2.progress.<task>.count``, such as
  the drops of an object, and restarts with its subtasks and uses once its completion no longer holds.
  :class:`~isaaclab_experimental.envs.MDP2RLEnv` records the count of every episode.
