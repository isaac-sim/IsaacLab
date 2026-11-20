.. _mdp2:

MDP2: Compiled MDPs
===================

.. note::

   MDP2 lives in ``isaaclab_experimental.mdp2``, and its tasks in ``isaaclab_tasks.mdp2``. It is an experimental
   feature: expect its API to change.

MDP2 describes a task's MDP (actions, observations, rewards, terminations, commands, events, and quantities) as
configuration, and compiles it into a few kernels per step. With the Warp backend, a whole environment step, physics
included, replays as one CUDA graph. :class:`~isaaclab_experimental.envs.MDP2RLEnv` runs an MDP2 task on Newton with
the Gym interface of :class:`~isaaclab.envs.ManagerBasedRLEnv`, so the RL wrappers work unchanged. MDP2 tasks run on
Newton only and do not import Isaac Sim.


Quick Start
-----------

The tasks ``Isaac-Cartpole-MDP2``, ``Isaac-Velocity-Flat-UnitreeGo2-MDP2``,
``Isaac-Velocity-Flat-WalkStand-UnitreeGo2-MDP2``, and ``Isaac-Lift-KukaAllegro-MDP2`` train with RSL-RL:

.. code-block:: bash

   uv run python scripts/reinforcement_learning/train.py --rl_library rsl_rl \
       --task Isaac-Cartpole-MDP2 physics=newton_mjwarp


Terms and uses
--------------

A term is a calculation with an implementation per backend: a batched Torch function, and a ``wp.func`` for one
world that MDP2 calls from the kernels it generates. A use binds a term to the data it reads and to its role in the
task. Inputs are named explicitly; nothing is gathered behind the term's back.

.. code-block:: python

   import warp as wp
   from isaaclab_experimental import mdp2


   @wp.func
   def _joint_vel_l1(world: int, joint_vel: wp.array2d(dtype=float), joint_ids: wp.array(dtype=wp.int32)) -> float:
       total = float(0.0)
       for k in range(joint_ids.shape[0]):
           total += wp.abs(joint_vel[world, joint_ids[k]])
       return total


   joint_vel_l1 = mdp2.Term(torch=lambda joint_vel, joint_ids: joint_vel[:, joint_ids].abs().sum(-1), warp=_joint_vel_l1)

   robot = mdp2.Entity("robot")
   cart_vel = mdp2.Reward(
       joint_vel_l1,
       inputs={"joint_vel": robot.joint_vel, "joint_ids": robot.joints("slider_to_cart")},
       weight=-0.01,
   )

The built-in terms are in :mod:`isaaclab_experimental.mdp2.mdp` and keep the names of their counterparts in
:mod:`isaaclab.envs.mdp`. A :class:`~isaaclab_experimental.mdp2.TaskCfg` groups the uses by role, and
:class:`~isaaclab_experimental.envs.MDP2RLEnvCfg` holds the task with its scene and simulation. See
``isaaclab_tasks/mdp2/cartpole`` for a complete task.

- **State.** A term can keep :class:`~isaaclab_experimental.mdp2.State` across steps, reset with its world's
  episode unless it is persistent. Other terms read it with :func:`~isaaclab_experimental.mdp2.term_state`.
- **Quantities.** A :class:`~isaaclab_experimental.mdp2.Quantity` computes a value per world once per step, which
  rewards, terminations, and events read as ``mdp2.quantities.<name>``.
- **Events** write scene fields in place. After the startup events, after the interval events in the worlds where
  they ran, and after all reset events, the scene makes the simulation see the written fields: joint and root state,
  fixed-base root poses, and gravity. It also does before a reset event that reads a field derived from written
  ones, such as body poses after joint writes, or that takes an entity whose fields were written; ``commit=True``
  forces it and ``commit=False`` skips it. Where a field is the solver's own state, as Newton's joint positions and
  root state usually are, the commit only tells the solver which worlds changed; otherwise it passes them through the
  asset's ``write_*_to_sim_mask`` methods, which also update what the asset derives from them, such as the previous
  joint velocities of joint accelerations. Fields that the simulation computes, such as body
  poses, contact forces, and joint accelerations, are read-only, and the builder rejects uses that write them.
- **Randomness** comes from per-world streams that each use owns (``Term(random=True)`` passes ``rng``), so the Torch
  and Warp backends draw the same numbers.


Building and inspecting
-----------------------

:class:`~isaaclab_experimental.mdp2.MDPBuilder` turns a task into an immutable
:class:`~isaaclab_experimental.mdp2.MDP`. Between adding the task and finalizing, the schedule can be inspected and
reordered within a phase. ``finalize`` reports every error at once, and ``mdp.describe()`` prints each op's layout and
each runtime function's steps.

.. code-block:: python

   builder = mdp2.MDPBuilder(scene_data.schema)
   builder.add_task(task)
   mdp = builder.finalize(device="cuda:0", backend="warp")
   print(mdp.describe())

:class:`~isaaclab_experimental.mdp2.MDPBuilder.replicate` runs different task configs, with their own rewards and
weights, in different worlds of one scene; :func:`~isaaclab_experimental.mdp2.split` does it from a task config.


Subtasks and progress
---------------------

A task may say when it is ``done``, and hold named subtasks, which are tasks too. Three functions combine tasks into
one:

.. list-table::
   :header-rows: 1

   * - Function
     - Subtasks run
     - The task is done
   * - :func:`~isaaclab_experimental.mdp2.compose`
     - in the same worlds, at the same steps
     - once ``require`` of them are (all by default)
   * - :func:`~isaaclab_experimental.mdp2.sequence`
     - in the same worlds, one after another
     - once the last one is
   * - :func:`~isaaclab_experimental.mdp2.split`
     - each in its own worlds
     - once the one of the world is
   * - :func:`~isaaclab_experimental.mdp2.choice`
     - one per episode and world, drawn at the reset
     - once the drawn one is

.. code-block:: python

   q = mdp2.quantities
   TASK = mdp2.compose(
       BASE,  # Uses of every step: observations, the action rate, the time limit
       mdp2.sequence(
           reach=mdp2.TaskCfg(rewards={"reach": ...}, done=q.near_object),
           grasp=mdp2.TaskCfg(rewards={"grasp": ...}, done=q.grasped, hold=3),
           lift=mdp2.Milestone(q.lifted),
       ),
   )
   terminations = {"success": mdp2.Termination(mdp.is_true, inputs={"value": mdp2.progress.done})}

- **Completion.** ``done`` is a field of one value per world, true where positive, that must hold for ``hold`` steps
  in a row, counted from the step after the task starts. A :class:`~isaaclab_experimental.mdp2.Milestone` is a task
  with only a completion. ``final=True`` (or ``False``) makes the parent done only while the subtask's completion
  still holds (or does not), and ``score`` weighs the subtask in the parent's score.
- **Stages.** A subtask of a sequence starts at the step after the previous one is done, and the last one keeps
  running. Its uses run only while it does: its rewards, terminations, commands, and events follow the subtask that
  ran when the step started, and its observations the one that runs after it. Starting restarts their state, timers,
  commands, and observation histories. Quantities are measurements: they run in every step, and only restart. Reset
  and startup events set up whole episodes, so they run at every reset; ``EventMode.START`` events run when their
  task starts, at the reset or during the episode, and at every reset for a task that always runs. Uses that the tasks define alike, but for their names, run as one
  while any of these tasks runs, so their state and histories carry on; a use of all of them is one of the sequence.
- **Repeats.** ``repeat=True`` counts a task's completions as ``mdp2.progress.<task>.count``, such as the drops of an
  object: once done, it restarts, with its subtasks and uses, when its completion no longer holds.
- **Names.** The uses of a subtask get its name as a prefix, as in ``rewards.grasp.grasp``, except in a split, whose
  subtasks run apart. A subtask reads its own fields by their names, as a task alone does, so each stage of a
  sequence can have its own ``reached``: its commands and quantities, the ``term_state`` and ``tunable`` values of its
  uses, and its ``mdp2.progress``; inner names win. Observations and actions of subtasks that take turns (a sequence,
  a choice, or a split) share the columns of their name, as wide as the widest, so a ``goal`` of each stage, or the
  ``joint_pos`` of two robots in two world configs, is one input of the policy. Those of subtasks that run together
  get columns of their own, named with the subtask's name, as in ``left.joint_pos``;
  :func:`~isaaclab_experimental.mdp2.rename` binds a part to other entities, so that it can run twice.
  :meth:`~isaaclab_experimental.mdp2.MDP.spec` names the columns, with joint and body names, and
  :meth:`~isaaclab_experimental.mdp2.MDP.action_mask` and :meth:`~isaaclab_experimental.mdp2.MDP.observation_mask`
  give those each world takes and writes.
- **Progress.** Every task's progress is a set of fields that terms read: ``mdp2.progress.done``,
  ``mdp2.progress.grasp.done``, and also ``holding``, ``active``, ``score``, ``done_at``, ``started_at``, and a
  sequence's ``stage`` (observed with :data:`~isaaclab_experimental.mdp2.mdp.one_hot`). ``MDP2RLEnv`` logs whether each
  task was done as ``Progress/<task>``.


Backends and capture
--------------------

- **Warp** fuses the terms' ``wp.func`` s into a few generated kernels per runtime function. It is the fast path.
- **Torch** runs each term as a batched Torch function. It is the reference, and convenient for prototyping.

A Warp MDP runs the Torch hooks of a term that has no Warp ones between its kernels, so a task can start in Torch and
move its expensive terms to Warp. Both backends, and such mixed MDPs, are captured: ``MDP2Env`` replays each step
after the first as CUDA graphs. Torch work in a graph runs on Warp's stream and allocates from a private memory pool,
so it must not copy data from the host or draw from Torch's random generators; use the term's ``rng`` instead.

**Python hooks** can do anything Python can. A hook can take scene entities bound as inputs, such as
``inputs={"robot": robot}``, and call any of their methods (writers, setters) or launch its own kernels, and it can
keep host objects in ``cache``. It runs inside the graphs unless the term is ``capturable=False``, for code that reads
values on the host: the environment then captures the work before and after it, physics included, and runs it
between the graphs. An event that writes through an entity's methods declares it as an output,
``outputs={"robot": robot}``, so that the builder orders and commits around it. ``Term(init=fn)`` fills a term's
state once at startup, on the host, with ``fn(ctx, **params)``.

**The step is a list.** ``MDP2EnvCfg.step`` holds the built-in work of a control step, and ``reset_step`` that of a
reset, which a step's ``Step.RESET`` and ``reset()`` run. Insert a :class:`~isaaclab_experimental.envs.Stage` anywhere
in them; it calls ``fn(env)`` with full access to the environment, inside the step's graph unless
``capturable=False``:

.. code-block:: python

   cfg.step                # [Step.PHYSICS, Step.RENDER, Step.EVALUATE, Step.RESET]
   cfg.reset_step          # [ResetStep.EVENTS, ResetStep.COMMIT, ResetStep.OBSERVE]

   cfg.step.insert(cfg.step.index(Step.PHYSICS) + 1, Stage(update_height_map))         # after the physics
   cfg.reset_step.insert(cfg.reset_step.index(ResetStep.OBSERVE), Stage(move_objects))  # after the commit
   cfg.step.append(Stage(log_episode_length, capturable=False))                         # at the end, on the host

The built-in steps must each appear once, in this order, except the optional ``Step.RENDER`` and
``ResetStep.RENDER``; the environment raises otherwise.

The fused ``wp.func`` path is what makes MDP2 fast: one kernel per statement instead of fused kernels costs Go2
78 → 121 µs and Lift 234 → 293 µs per step at 4096 worlds, also inside a graph. Library terms use it; Python hooks and
stages are for what does not fit it.

A captured step replays what was recorded, so three things are fixed after the first steps, without an error if
they change:

- **Python in capturable hooks** runs once, during the capture: a branch on a host value takes the same path on every
  replay.
- **Params** are constants of the kernels and hooks: changing ``params`` has no effect.
- **Buffers** are bound by address: write the MDP's arrays in place, such as ``state.reward_weights.torch[:] = ...``,
  rather than assigning new ones.

Values that change while training, such as a curriculum's difficulty, belong in term state or quantities, which
terms read on the device: the Lift task's curriculum keeps its progress as the state of its ``adr`` event.

``MDP2EnvCfg.verify_capture = 20`` checks a task for such mistakes: before building, it steps the task uncaptured and
captured from the same seed and actions, on reproducible physics where the backend offers it, and raises at the first
value that differs, naming it and the Python code that runs inside the graphs.

Changing the physics model inside a graph, such as armatures in a reset event, makes every replayed step refresh the
solver's model: about 250 µs per step for 4096 Cartpole worlds on MJWarp. Randomize the model at startup where you
can.


Cameras, composition, and evaluation
------------------------------------

These features let an evaluation framework, such as Isaac Lab Arena, build its environments on MDP2: a policy such as
a vision-language-action model reads camera images and named observation terms, and the environment reports how each
episode ended.

- **Cameras** are scene entities whose data types are fields, such as ``external_camera.rgb``. ``MDP2Env`` renders
  the cameras that terms read once per control step at ``Step.RENDER``, after the physics, inside the step's CUDA
  graph; a step without ``Step.RENDER`` is an error when terms read cameras. Insert ``ResetStep.RENDER`` before
  ``ResetStep.OBSERVE`` in ``reset_step`` to render them again after the resets, so that new episodes start with
  their own images.
- **Observation groups.** A group in :attr:`~isaaclab_experimental.mdp2.TaskCfg.observations` is a dict of uses,
  concatenated into one ``float32`` tensor, or an :class:`~isaaclab_experimental.mdp2.ObservationGroup`. With
  ``concatenate=False``, the group is observed as a dict of tensors with each term's own shape and type, such as
  ``uint8`` images from :data:`~isaaclab_experimental.mdp2.mdp.image_rgb`.
- **Action terms.** A :class:`~isaaclab_experimental.mdp2.DirectAction` scales its columns of the action into a
  field. An :class:`~isaaclab_experimental.mdp2.Action` runs a term's hook on its columns instead, such as
  :data:`~isaaclab_experimental.mdp2.mdp.binary_joint_position` for a gripper, which closes on either side of a
  threshold, as MDP1's binary joint actions do.
- **Composition.** :func:`~isaaclab_experimental.mdp2.compose` merges partial task configs term by term, such as
  those of a scene, an embodiment, and a task, and reports uses that two of them define; a later part removes a use
  by setting it to ``None``. Uses keep the order of the parts.
- **Success** is the completion of the task, built from milestones and subtasks (see `Subtasks and progress`_),
  which a :class:`~isaaclab_experimental.mdp2.Termination` ends the episode on.
- **Episode budget.** With ``episode_budget``, ``MDP2RLEnv`` runs that many episodes over its worlds, then stops them,
  and records each episode on the device: its world, length, how it ended, its ``Quantity(record=True)`` values, and
  the step at which each task was done. :meth:`~isaaclab_experimental.envs.MDP2RLEnv.episode_records` returns them.

.. code-block:: python

   cfg.episode_budget = 50
   env = MDP2RLEnv(cfg)
   obs, _ = env.reset()
   while env.episodes_completed < cfg.episode_budget:
       obs, *_ = env.step(policy(obs["policy"], obs["camera_obs"]))
   success_rate = env.episode_records()["termination/success"].mean()


Changing a task while it trains
-------------------------------

The values and structure of a task can change while it runs, at the cost of what they invalidate.

.. code-block:: python

   track = mdp2.Reward(mdp.track_lin_vel_xy_exp, ..., params={"std": mdp2.Tunable(0.5)}, weight=mdp2.Tunable(1.5))
   rewards = {"track": track}
   env.tune("rewards.track.weight", 3.0)                              # In each world at its next reset
   env.tune("rewards.track.std", 0.25, worlds=slice(0, 2048), at="step")  # From the next step

   rewards = {"stand_still": mdp2.Reward(..., when=mdp2.commands.standing)}  # Only where the mode command says so
   env.update_task(new_task)                                           # Recompile on the same scene, keep the state

- **Runtime values.** A param declared :class:`~isaaclab_experimental.mdp2.Tunable` is a field of one value per
  world, ``mdp2.tunable(use, name)``, which ``wp.func`` s take as their world's value and other terms and events may
  read and write, such as a curriculum; a tunable reward weight is the reward's column of ``state.reward_weights``.
  :meth:`~isaaclab_experimental.envs.MDP2Env.tune` changes either in place while graphs replay, at each world's next
  reset by default, so that no episode mixes two values, and leaves the other values alone; each world counts the
  tunings that took effect in ``state.revision``, which episode records hold. Tasks without tunable values compile
  to the same kernels.
- **Switches.** ``when`` on a reward, termination, observation, command, or interval event runs it where a field was
  positive at the start of the step, and restarts it where it turns on; ``when=mdp2.Tunable(True)`` is a switch per
  world that ``tune`` flips.
- **Multi-task learning.** :func:`~isaaclab_experimental.mdp2.choice` draws one subtask per episode and world, by
  weights; ``Progress/<task>`` logs each over the worlds that drew it.
- **Variations.** :func:`~isaaclab_experimental.mdp2.mdp.variation` draws a value per world at every reset, writes it
  to the scene, and records it per episode: :class:`~isaaclab_experimental.mdp2.mdp.Uniform` in a tunable range,
  :class:`~isaaclab_experimental.mdp2.mdp.Choice`, or :class:`~isaaclab_experimental.mdp2.mdp.Replay` of recorded
  values in the order of the resets, which does not depend on the number of worlds.
- **Actions at two rates.** ``rate=ActionRate.CONTROL`` applies an action once per control step, before the physics
  steps, such as an embedded whole-body controller that writes joint targets; the default applies it before every
  physics step. Tasks of a sequence may take actions of their own.
- **New structure.** :meth:`~isaaclab_experimental.envs.MDP2Env.update_task` builds another task on the same scene
  and carries over, with :func:`~isaaclab_experimental.mdp2.migrate`, the state, random streams, and tunable values
  of the uses that both have, while other weights and params are the new task's; it truncates the episodes by
  default, and the next step compiles and captures anew.

Measured on Go2 at 4096 worlds (MDP time per step, physics replaced by nothing): the plain task 81 µs, every reward
weight and tracking width tunable 81 µs, all rewards switched 86 µs, the rewards in two stages of a sequence 103 µs.


Porting from Arena
------------------

.. list-table::
   :header-rows: 1

   * - Arena
     - MDP2
   * - ``CompletionCriteria`` with subtasks
     - a task with ``done`` or milestones, combined with ``compose`` or ``sequence``
   * - ``subtasks_are_sequential``
     - :func:`~isaaclab_experimental.mdp2.sequence` rather than :func:`~isaaclab_experimental.mdp2.compose`
   * - ``K`` of the subtasks
     - ``compose(..., require=K)``
   * - ``desired_subtask_success_state``
     - ``final=True`` or ``False`` on the subtask
   * - ``TrueForConsecutiveSteps``
     - ``hold=n``
   * - ``object_on_destination``
     - :data:`~isaaclab_experimental.mdp2.mdp.resting_on`, held with ``hold``
   * - tracked events
     - ``repeat=True``, counted as ``progress.<task>.count``
   * - run-time variations and their JSONL records
     - :func:`~isaaclab_experimental.mdp2.mdp.variation`, with ``episode_budget`` records
   * - per-episode results
     - ``env.episode_records()``: ``progress/<task>`` (the step it was done), ``progress/score``, recorded quantities
   * - configclass merging of scene, embodiment, and task
     - :func:`~isaaclab_experimental.mdp2.compose`, and :func:`~isaaclab_experimental.mdp2.rename` for a part twice


Limitations
-----------

- Newton only. Different robots in different worlds are not supported, since Newton's articulation views need
  homogeneous worlds; different task configs on the same robot are.
- The environments support actuators that CUDA graphs can replay: Isaac Lab actuators with
  :attr:`~isaaclab.actuators.ActuatorBase.capturable`, and Newton's own actuators, which run delayed and BAM ones.
  Actuator networks are not supported.
- Joint and body orderings (``ArticulationCfg.joint_ordering`` and ``body_ordering``) are not supported: terms read
  the asset's data in kernels, and orderings make the asset keep reordered copies that it refreshes outside the
  graphs.
- Rendering runs on the host after each control step, outside the CUDA graphs: at most once per control step, when
  one of its physics steps reaches ``sim.render_interval``, and after the step's resets (MDP1 renders inside the
  decimation loop, before resets). ``render_enabled`` and ``num_rerenders_on_reset`` are not supported.
- A step that replays as a CUDA graph does the host work of its physics steps once, after the replay
  (:meth:`~isaaclab.sim.SimulationContext.after_replay`): it counts them, advances the simulation time, and waits
  while a visualizer pauses training. The solver's post-step host checks do not run.
- Asset data that is derived on demand, such as ``root_lin_vel_b``, is not refreshed by a captured step. Terms read
  the fields of the scene data, which alias the solver state; code outside the MDP that reads asset data between
  steps should use those too, for example ``root_com_vel_w`` rotated by ``root_link_pose_w``.
