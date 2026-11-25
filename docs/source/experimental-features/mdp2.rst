.. _mdp2:

MDP2: Compiled MDPs
===================

.. note::

   MDP2 lives in ``isaaclab_experimental.mdp2``, and its tasks in ``isaaclab_tasks_experimental.mdp2``. It is an
   experimental feature: expect its API to change.

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
``isaaclab_tasks_experimental/mdp2/cartpole`` for a complete task.

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
weights, in different worlds of one scene.


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

   cfg.step                # [Step.PHYSICS, Step.EVALUATE, Step.RESET]
   cfg.reset_step          # [ResetStep.EVENTS, ResetStep.COMMIT, ResetStep.OBSERVE]

   cfg.step.insert(cfg.step.index(Step.PHYSICS) + 1, Stage(update_height_map))         # after the physics
   cfg.reset_step.insert(cfg.reset_step.index(ResetStep.OBSERVE), Stage(move_objects))  # after the commit
   cfg.step.append(Stage(log_episode_length, capturable=False))                         # at the end, on the host

The built-in steps must each appear once, in this order; the environment raises otherwise.

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


Limitations
-----------

- Newton only. Different robots in different worlds are not supported, since Newton's articulation views need
  homogeneous worlds; different task configs on the same robot are.
- The environments support implicit, ideal PD, and DC motor actuators only. Captured steps replay the actuators'
  masked resets, and delayed actuators draw their delays from Torch's random generator on reset, which a replayed
  graph would repeat. Actuator-network and BAM actuators are not supported yet either.
- Joint and body orderings (``ArticulationCfg.joint_ordering`` and ``body_ordering``) are not supported: terms read
  the asset's data in kernels, and orderings make the asset keep reordered copies that it refreshes outside the
  graphs.
- Rendering runs on the host after each control step, outside the CUDA graphs: at most once per control step, when
  one of its physics steps reaches ``sim.render_interval``, and after the step's resets (MDP1 renders inside the
  decimation loop, before resets). ``render_enabled`` and ``num_rerenders_on_reset`` are not supported.
- A step that replays as a CUDA graph skips the host bookkeeping of ``SimulationContext.step``: the physics step
  count and the simulation time stop advancing after the first steps, a visualizer that pauses training does not
  block the training loop, and the solver's post-step host checks do not run. Consumers that key frames by the
  physics step count, such as video recorders, are not supported.
- Asset data that is derived on demand, such as ``root_lin_vel_b``, is not refreshed by a captured step. Terms read
  the fields of the scene data, which alias the solver state; code outside the MDP that reads asset data between
  steps should use those too, for example ``root_com_vel_w`` rotated by ``root_link_pose_w``.
