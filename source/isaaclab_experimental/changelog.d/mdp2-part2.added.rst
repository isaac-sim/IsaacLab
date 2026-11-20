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
  while it does, and restart when it starts. A subtask reads its own fields by name, as a task that runs alone does:
  its commands, quantities, term state, tunable values, and progress. Terms read the progress of every task, such as
  ``mdp2.progress.grasp.done``, and :class:`~isaaclab_experimental.envs.MDP2RLEnv` logs it as ``Progress/<task>`` and
  records the step at which each task was done.
* Added :meth:`~isaaclab_experimental.mdp2.MDP.spec`, which names the columns of the action and of the observation
  buffers, with the names of their elements, such as joint names, and
  :meth:`~isaaclab_experimental.mdp2.MDP.action_mask` and :meth:`~isaaclab_experimental.mdp2.MDP.observation_mask`,
  which give the columns each world takes and writes.
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
* Added actions to the tasks of an MDP2 sequence: each task writes its actions while it runs, and actions of the same
  name share their columns of the action, so that one policy drives every task.
* Added :class:`~isaaclab_experimental.mdp2.ActionRate` to MDP2 actions: an action at ``ActionRate.CONTROL`` applies
  once per control step, before the physics steps, such as an embedded whole-body controller or inverse kinematics,
  and one at ``ActionRate.PHYSICS``, the default, before every physics step.
  :func:`~isaaclab_experimental.mdp2.process_actions` runs the former.
* Added nested splits to MDP2: a :func:`~isaaclab_experimental.mdp2.split` may be a step of a sequence, and several
  subtasks that split divide the same worlds, such as robots and terrains, each world running the subtask of its
  share of each.
* Added :func:`~isaaclab_experimental.mdp2.choice`, which draws one subtask per episode and world at the reset, by
  weights, for multi-task learning: only the drawn subtask's uses run, and ``Progress/<task>`` logs its success over
  the worlds that drew it.
* Added :func:`~isaaclab_experimental.mdp2.rename`, which binds a part of a task to other scene entities, so that a
  part can run twice, such as two arms. Uses of subtasks that run together get columns of their own, named with the
  subtask's name, as in ``left.joint_pos``; uses of subtasks that take turns share the columns of their name.
* Added start events to MDP2, ``EventMode.START``: an event of a task of a sequence or a choice, or of a task that
  repeats, writes the scene when its task starts, at the reset or during the episode, where the scene commits it.
* Added :meth:`~isaaclab_experimental.envs.MDP2Env.update_task`, which replaces the task of a running environment on
  the same scene, such as with a reward added, and :func:`~isaaclab_experimental.mdp2.migrate`, which carries over the
  state, random streams, and reward weights of the uses that both tasks have.
* Added variations to the MDP2 library, :func:`~isaaclab_experimental.mdp2.mdp.variation`: a reset event that draws a
  value per world, such as how far a drawer is open, writes it to a scene field, and records it per episode, drawn
  :class:`~isaaclab_experimental.mdp2.mdp.Uniform` (in a tunable range), from a
  :class:`~isaaclab_experimental.mdp2.mdp.Choice`, or replayed in the order of the resets with
  :class:`~isaaclab_experimental.mdp2.mdp.Replay`, independent of the number of worlds.
