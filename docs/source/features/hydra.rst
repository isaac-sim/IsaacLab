Hydra Configuration System
==========================

.. currentmodule:: isaaclab

Isaac Lab supports the `Hydra <https://hydra.cc/docs/intro/>`_ configuration system to modify the task's
configuration using command line arguments, which can be useful to automate experiments and perform hyperparameter tuning.

Any parameter of the environment can be modified by adding one or multiple elements of the form ``env.a.b.param1=value``
to the command line input, where ``a.b.param1`` reflects the parameter's hierarchy, for example ``env.actions.joint_effort.scale=10.0``.
Similarly, the agent's parameters can be modified by using the ``agent`` prefix, for example ``agent.seed=2024``.

The way these command line arguments are set follow the exact structure of the configuration files. Since the different
RL frameworks use different conventions, there might be differences in the way the parameters are set. For example,
with *rl_games* the seed will be set with ``agent.params.seed``, while with *rsl_rl*, *skrl* and *sb3* it will be set with
``agent.seed``.

As a result, training with hydra arguments can be run with the following syntax:

.. tab-set::
    :sync-group: rl-train

    .. tab-item:: rsl_rl
        :sync: rsl_rl

        .. code-block:: shell

            python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Cartpole-v0 --headless env.actions.joint_effort.scale=10.0 agent.seed=2024

    .. tab-item:: rl_games
        :sync: rl_games

        .. code-block:: shell

            python scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Cartpole-v0 --headless env.actions.joint_effort.scale=10.0 agent.params.seed=2024

    .. tab-item:: skrl
        :sync: skrl

        .. code-block:: shell

            python scripts/reinforcement_learning/skrl/train.py --task=Isaac-Cartpole-v0 --headless env.actions.joint_effort.scale=10.0 agent.seed=2024

    .. tab-item:: sb3
        :sync: sb3

        .. code-block:: shell

            python scripts/reinforcement_learning/sb3/train.py --task=Isaac-Cartpole-v0 --headless env.actions.joint_effort.scale=10.0 agent.seed=2024

The above command will run the training script with the task ``Isaac-Cartpole-v0`` in headless mode, and set the
``env.actions.joint_effort.scale`` parameter to 10.0 and the ``agent.seed`` parameter to 2024.

.. note::

    To keep backwards compatibility, and to provide a more user-friendly experience, we have kept the old cli arguments
    of the form ``--param``, for example ``--num_envs``, ``--seed``, ``--max_iterations``. These arguments have precedence
    over the hydra arguments, and will overwrite the values set by the hydra arguments.


Modifying advanced parameters
-----------------------------

Callables
^^^^^^^^^

It is possible to modify functions and classes in the configuration files by using the syntax ``module:attribute_name``.
For example, in the Cartpole environment:

.. literalinclude:: ../../../source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/cartpole_env_cfg.py
    :language: python
    :start-at: class ObservationsCfg
    :end-at: policy: PolicyCfg = PolicyCfg()
    :emphasize-lines: 9

we could modify ``joint_pos_rel`` to compute absolute positions instead of relative positions with
``env.observations.policy.joint_pos_rel.func=isaaclab.envs.mdp:joint_pos``.

Setting parameters to None
^^^^^^^^^^^^^^^^^^^^^^^^^^

To set parameters to None, use the ``null`` keyword, which is a special keyword in Hydra that is automatically converted to None.
In the above example, we could also disable the ``joint_pos_rel`` observation by setting it to None with
``env.observations.policy.joint_pos_rel=null``.

Dictionaries
^^^^^^^^^^^^
Elements in dictionaries are handled as a parameters in the hierarchy. For example, in the Cartpole environment:

.. literalinclude:: ../../../source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/cartpole_env_cfg.py
    :language: python
    :lines: 90-114
    :emphasize-lines: 11

the ``position_range`` parameter can be modified with ``env.events.reset_cart_position.params.position_range="[-2.0, 2.0]"``.
This example shows two noteworthy points:

- The parameter we set has a space, so it must be enclosed in quotes.
- The parameter is a list while it is a tuple in the config. This is due to the fact that Hydra does not support tuples.


Modifying inter-dependent parameters
------------------------------------

Particular care should be taken when modifying the parameters using command line arguments. Some of the configurations
perform intermediate computations based on other parameters. These computations will not be updated when the parameters
are modified.

For example, for the configuration of the Cartpole camera depth environment:

.. literalinclude:: ../../../source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/cartpole_camera_env.py
    :language: python
    :start-at: class CartpoleDepthCameraEnvCfg
    :end-at: tiled_camera.width
    :emphasize-lines: 10, 15

If the user were to modify the width of the camera, i.e. ``env.tiled_camera.width=128``, then the parameter
``env.observation_space=[80,128,1]`` must be updated and given as input as well.

Similarly, the ``__post_init__`` method is not updated with the command line inputs. In the ``LocomotionVelocityRoughEnvCfg``, for example,
the post init update is as follows:

.. literalinclude:: ../../../source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py
    :language: python
    :start-at: class LocomotionVelocityRoughEnvCfg
    :emphasize-lines: 23, 29, 31

Here, when modifying ``env.decimation`` or ``env.sim.dt``, the user needs to give the updated ``env.sim.render_interval``,
``env.scene.height_scanner.update_period``, and ``env.scene.contact_forces.update_period`` as input as well.
