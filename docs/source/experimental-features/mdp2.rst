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
       --task Isaac-Cartpole-MDP2 presets=newton_mjwarp


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
- **Events** write scene fields in place. Before an event with ``commit=True``, and after all reset events, the scene
  makes the simulation see the written fields: joint and root state, fixed-base root poses, and gravity.
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

**Eager terms** (``Term(eager=fn)``) are plain Python functions with MDP1-style access to the scene, called as
``fn(ctx, **params)``. They may synchronize with the host or call any library, so they never run inside a graph:
the environment captures the work before and after them, physics included, and runs them between the graphs.
``Term(init=fn)`` fills a term's state once at startup in the same way.


Limitations
-----------

- Newton only. Different robots in different worlds are not supported, since Newton's articulation views need
  homogeneous worlds; different task configs on the same robot are.
- Asset resets with a mask reset the actuator state of every world, so the environments support implicit, ideal PD,
  and DC motor actuators only, which keep no state, and refuse delayed, actuator-network, and BAM actuators.
- Joint orderings (``ArticulationCfg.joint_ordering``) are not supported: terms see the solver's joint order.
- Asset data that is derived on demand, such as ``root_lin_vel_b``, is not refreshed by a captured step. Terms read
  the fields of the scene data, which alias the solver state; code outside the MDP that reads asset data between
  steps should use those too, for example ``root_com_vel_w`` rotated by ``root_link_pose_w``.
