Optimize Stage Creation
=======================

Isaac Lab supports two experimental features to speed-up stage creation: **fabric cloning** and **stage in memory**.
These features are particularly effective for large-scale RL setups with thousands of environments.

What These Features Do
-----------------------

**Fabric Cloning**

- Clones environments using Fabric library (see `USD Fabric USDRT Documentation <https://docs.omniverse.nvidia.com/kit/docs/usdrt.scenegraph/latest/usd_fabric_usdrt.html>`_)
- Partially supported and enabled by default on some environments (see `Limitations`_ section for a list)

**Stage in Memory**

- Constructs the stage in memory, rather than with a USD file, avoiding overhead from disk I/O
- After stage creation, if rendering is required, the stage is attached to the USD context, returning to the default stage configuration
- Not enabled by default

Usage Examples
--------------

Fabric cloning can be toggled by setting the :attr:`isaaclab.scene.InteractiveSceneCfg.clone_in_fabric` flag.
For a full guide on the template-based cloning system, see :doc:`cloning`.

**Using Fabric Cloning with a RL environment**

.. code-block:: python

    # create environment configuration
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.clone_in_fabric = True
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)


Stage in memory can be toggled by setting the :attr:`isaaclab.sim.SimulationCfg.create_stage_in_memory` flag.

**Using Stage in Memory with a RL environment**

.. code-block:: python

    # create config and set flag
    cfg = CartpoleEnvCfg()
    cfg.scene.num_envs = 1024
    cfg.sim.create_stage_in_memory = True
    # create env with stage in memory
    env = ManagerBasedRLEnv(cfg=cfg)

When using stage in memory without an existing RL environment class, wrap the stage creation steps
in a ``with`` statement to set the stage context. The stage is automatically attached
to the USD context when ``SimulationContext`` is created with ``create_stage_in_memory=True``.

**Using Stage in Memory with a manual scene setup**

.. code-block:: python

    # init simulation context with stage in memory
    # Note: stage is automatically attached to USD context
    sim = SimulationContext(cfg=SimulationCfg(create_stage_in_memory=True))

    # grab stage and set stage context
    with stage_utils.use_stage(sim.stage):
        # create cartpole scene
        scene_cfg = CartpoleSceneCfg(num_envs=1024)
        scene = InteractiveScene(scene_cfg)

    sim.play()


Limitations
-----------

**Fabric Cloning**

- Fabric-cloned environments must be accessed using USDRT functions, rather than USD functions.
- Fabric cloning is partially supported and enabled by default on some environments, listed here.

.. code-block:: none

    1.  Isaac-Ant-Direct
    2.  Isaac-Ant
    3.  Isaac-Cartpole-Direct
    4.  IsaacContrib-Cartpole-Showcase-Direct
    5.  Isaac-Cartpole
    20. IsaacContrib-Factory-GearMesh-Direct
    21. IsaacContrib-Factory-NutThread-Direct
    22. IsaacContrib-Factory-PegInsert-Direct
    23. Isaac-Open-Drawer-Franka-Direct
    24. Isaac-Humanoid-Direct
    25. Isaac-Humanoid
    26. Isaac-Quadcopter-Direct-v0
    27. Isaac-Reorient-Cube-Allegro-Direct
    28. Isaac-Reorient-Cube-Allegro
    29. Isaac-Reorient-Cube-Shadow-Direct
    30. Isaac-Reorient-Cube-Shadow-OpenAI-FF-Direct
    31. Isaac-Reorient-Cube-Shadow-OpenAI-LSTM-Direct

**Stage in Memory**

- Cannot be currently enabled at the same time as **Fabric Cloning**.

- The stage is automatically attached to the USD context at ``SimulationContext`` creation, ensuring proper
  lifecycle events for viewport and physics systems.

- Certain low-level Kit APIs do not yet support stage in memory.

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
