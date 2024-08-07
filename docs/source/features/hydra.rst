Hydra Configuration System
==========================

.. currentmodule:: omni.isaac.lab

Isaac Lab supports the `Hydra <https://hydra.cc/docs/intro/>`_ system to modify the task's
configuration using command line arguments. This can be useful to automate experiments and hyperparameter tuning.
The parameters can be modified by using the following syntax:

.. tab-set::
    :sync-group: rl-train

    .. tab-item:: rsl_rl
        :sync: rsl_rl

        .. code-block:: shell

            python source/standalone/workflows/rsl_rl/train_hydra.py --task=Isaac-Cartpole-v0 --headless env.actions.joint_effort.scale=10.0 agent.seed=2024

    .. tab-item:: rl_games
        :sync: rl_games

        .. code-block:: shell

            python source/standalone/workflows/rl_games/train_hydra.py --task=Isaac-Cartpole-v0 --headless env.actions.joint_effort.scale=10.0 agent.params.seed=2024

    .. tab-item:: skrl
        :sync: skrl

        .. code-block:: shell

            python source/standalone/workflows/skrl/train_hydra.py --task=Isaac-Cartpole-v0 --headless env.actions.joint_effort.scale=10.0 agent.seed=2024

The above command will run the training script with the task ``Isaac-Cartpole-v0`` in headless mode, and set the
``env.actions.joint_effort.scale`` parameter to 10.0 and the ``agent.seed`` parameter to 2024.

.. note::

    The environment specific parameters are prefixed with ``env`` and the agent specific parameters with ``agent``.


.. attention::

    Particular care should be taken when modifying the parameters using command line arguments. Some of the configurations
    perform intermediate computations based on other parameters. These computations will not be updated when the parameters
    are modified.

    For example, for the configuration of the Cartpole camera depth environment:

    .. literalinclude:: ../../../source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/cartpole/cartpole_camera_env.py
        :language: python
        :start-at: class CartpoleDepthCameraEnvCfg
        :end-at: tiled_camera.width
        :emphasize-lines: 16

    If the user were to modify the width of the camera, i.e. ``env.tiled_camera.with=128``, then the parameter
    ``env.num_observations=10240`` (1*80*128) must be updated and given as input as well.

    Similarly, the ``__post_init__`` method is not updated with the command line inputs. In the ``LocomotionVelocityRoughEnvCfg``, for example,
    the post init update is as follows:

    .. literalinclude:: ../../../source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py
        :language: python
        :start-at: class LocomotionVelocityRoughEnvCfg
        :emphasize-lines: 23, 29, 31

    Here, when modifying ``env.decimation`` or ``env.sim.dt``, the user would have to manually update ``env.sim.render_interval``,
    ``env.scene.height_scanner.update_period``, and ``env.scene.contact_forces.update_period`` as well.
