Optimize Stage Creation
=======================

Isaac Lab supports two experimental features to speed-up stage creation: **fabric cloning** and **stage in memory**.
These features are particularly effective for large-scale RL setups with thousands of environments.

What These Features Do
-----------------------

**Fabric Cloning**

- Clones environments using Fabric library (see `USD Fabric USDRT Documentation <https://docs.omniverse.nvidia.com/kit/docs/usdrt/latest/docs/usd_fabric_usdrt.html>`_)
- Partially supported and enabled by default on some environments (see `Limitations`_ section for a list)

**Stage in Memory**

- Constructs the stage in memory, rather than with a USD file, avoiding overhead from disk I/O
- After stage creation, if rendering is required, the stage is attached to the USD context, returning to the default stage configuration
- Not enabled by default

Usage Examples
--------------

Fabric cloning can be toggled by setting the ``clone_in_fabric`` flag in the ``InteractiveSceneCfg`` configuration.

**Using Fabric Cloning with a RL environment**

.. code-block:: python

    # create environment configuration
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.clone_in_fabric = True
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)


Stage in memory can be toggled by setting the ``create_stage_in_memory`` in the ``SimulationCfg`` configuration.

**Using Stage in Memory with a RL environment**

.. code-block:: python

    # create config and set flag
    cfg = CartpoleEnvCfg()
    cfg.scene.num_envs = 1024
    cfg.sim.create_stage_in_memory = True
    # create env with stage in memory
    env = ManagerBasedRLEnv(cfg=cfg)

Note, if stage in memory is enabled without using an existing RL environment class, a few more steps are need.
The stage creation steps should be wrapped in a ``with`` statement to set the stage context.
If the stage needs to be attached, the ``attach_stage_to_usd_context`` function should be called after the stage is created.

**Using Stage in Memory with a manual scene setup**

.. code-block:: python

    # init simulation context with stage in memory
    sim = SimulationContext(cfg=SimulationCfg(create_stage_in_memory=True))

    # grab stage in memory and set stage context
    stage_in_memory = sim.get_initial_stage()
    with stage_utils.use_stage(stage_in_memory):
        # create cartpole scene
        scene_cfg = CartpoleSceneCfg(num_envs=1024)
        scene = InteractiveScene(scene_cfg)
        # attach stage to memory after stage is created
        sim_utils.attach_stage_to_usd_context()

    sim.play()


Limitations
-----------

**Fabric Cloning**

- Fabric-cloned environments must be accessed using USDRT functions, rather than USD functions.
- Fabric cloning is partially supported and enabled by default on some environments, listed here.

.. code-block:: none

    1.  Isaac-Ant-Direct-v0
    2.  Isaac-Ant-v0
    3.  Isaac-Cartpole-Direct-v0
    4.  Isaac-Cartpole-Showcase-Box-Box-Direct-v0
    5.  Isaac-Cartpole-Showcase-Box-Discrete-Direct-v0
    6.  Isaac-Cartpole-Showcase-Box-MultiDiscrete-Direct-v0
    7.  Isaac-Cartpole-Showcase-Dict-Box-Direct-v0
    8.  Isaac-Cartpole-Showcase-Dict-Discrete-Direct-v0
    9.  Isaac-Cartpole-Showcase-Dict-MultiDiscrete-Direct-v0
    10. Isaac-Cartpole-Showcase-Discrete-Box-Direct-v0
    11. Isaac-Cartpole-Showcase-Discrete-Discrete-Direct-v0
    12. Isaac-Cartpole-Showcase-Discrete-MultiDiscrete-Direct-v0
    13. Isaac-Cartpole-Showcase-MultiDiscrete-Box-Direct-v0
    14. Isaac-Cartpole-Showcase-MultiDiscrete-Discrete-Direct-v0
    15. Isaac-Cartpole-Showcase-MultiDiscrete-MultiDiscrete-Direct-v0
    16. Isaac-Cartpole-Showcase-Tuple-Box-Direct-v0
    17. Isaac-Cartpole-Showcase-Tuple-Discrete-Direct-v0
    18. Isaac-Cartpole-Showcase-Tuple-MultiDiscrete-Direct-v0
    19. Isaac-Cartpole-v0
    20. Isaac-Factory-GearMesh-Direct-v0
    21. Isaac-Factory-NutThread-Direct-v0
    22. Isaac-Factory-PegInsert-Direct-v0
    23. Isaac-Franka-Cabinet-Direct-v0
    24. Isaac-Humanoid-Direct-v0
    25. Isaac-Humanoid-v0
    26. Isaac-Quadcopter-Direct-v0
    27. Isaac-Repose-Cube-Allegro-Direct-v0
    28. Isaac-Repose-Cube-Allegro-NoVelObs-v0
    29. Isaac-Repose-Cube-Allegro-v0
    30. Isaac-Repose-Cube-Shadow-Direct-v0
    31. Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0
    32. Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0

**Stage in Memory**

- Cannot be currently enabled at the same time as **Fabric Cloning**.

- Attaching stage in memory to the USD context can be slow, offsetting some or all of the performance benefits.

  - Note, attaching is only necessary when rendering is enabled. For example, in headless mode, attachment is not required.

- Certain low-level Kit APIs do not yet support stage in memory.

  - In most cases, when these APIs are hit, existing scripts will automatically early attach the stage and print a warning message.
  - In one particular case, for some environments, the API call to color the ground plane is skipped, when stage in memory is enabled.


Benchmark Results
-----------------

Performance comparison cloning 4000 ShadowHand robots with rendering enabled

+--------+-----------------+-------------------+------------------------+---------------------------+------------------------+------------------------+
| Test # | Stage in Memory | Clone in Fabric   | Attach Stage Time (s)  | Fabric Attach Time (s)    | Clone Paths Time (s)   | First Step Time (s)    |
+========+=================+===================+========================+===========================+========================+========================+
| 1      | Yes             | Yes               | 3.88                   | 0.15                      | 4.84                   | 1.39                   |
+--------+-----------------+-------------------+------------------------+---------------------------+------------------------+------------------------+
| 2      | No              | No                | —                      | 60.17                     | 4.46                   | 3.52                   |
+--------+-----------------+-------------------+------------------------+---------------------------+------------------------+------------------------+
| 3      | No              | Yes               | —                      | 0.47                      | 4.72                   | 2.56                   |
+--------+-----------------+-------------------+------------------------+---------------------------+------------------------+------------------------+
| 4      | Yes             | No                | 42.64                  | 21.75                     | 1.87                   | 2.16                   |
+--------+-----------------+-------------------+------------------------+---------------------------+------------------------+------------------------+
