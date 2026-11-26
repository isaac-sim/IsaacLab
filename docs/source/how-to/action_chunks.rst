:orphan:

Applying action chunks
======================

Policies that plan action chunks, such as GR00T, ACT and diffusion policies, output the next several
commands per call. Without chunks, every joint action term applies one command on all ``decimation``
physics steps of an environment step, so a runner has to step the environment once per command and
drop the rest of the chunk when an environment resets.

Set ``chunk_length`` on a joint action term and the policy acts once per environment step instead.
The action holds ``chunk_length`` commands per joint, ordered by command, then by joint. Each command
applies for ``decimation / chunk_length`` physics steps, and ``chunk_length`` must divide ``decimation``.

.. code-block:: python

   from isaaclab.envs import ManagerBasedRLEnvCfg
   from isaaclab.envs.mdp import JointEffortActionCfg, JointPositionActionCfg
   from isaaclab.utils import configclass


   @configclass
   class ActionsCfg:
       arm = JointPositionActionCfg(asset_name="robot", joint_names=[".*"], chunk_length=16)
       feedforward = JointEffortActionCfg(asset_name="robot", joint_names=[".*"], chunk_length=16)


   @configclass
   class MyEnvCfg(ManagerBasedRLEnvCfg):
       actions: ActionsCfg = ActionsCfg()
       # 16 commands per action, each held for 4 physics steps
       decimation = 16 * 4

With ``num_joints`` joints, the position term takes ``16 * num_joints`` values per environment. The
environment steps once per chunk:

.. code-block:: python

   obs, reward, terminated, truncated, info = env.step(policy(obs))

To get smooth motion from a low-rate policy, set ``interpolate=True`` on
:class:`~isaaclab.envs.mdp.actions.JointPositionActionCfg`. The action is then one target per joint,
reached in ``chunk_length`` equal increments from the previous target, or from the current joint
positions after a reset.

Limits
------

- Observations and rewards are computed once per environment step, that is once per chunk. A task that
  needs them after every command should keep one command per environment step.
- :class:`~isaaclab.envs.mdp.actions.RelativeJointPositionActionCfg` adds its action to the joint
  positions on every physics step, so it does not support ``chunk_length`` above one.
- Chunks apply to joint position, velocity and effort actions. Task-space actions such as differential
  inverse kinematics are not supported.
