.. _isaac-lab-quickstart:

Quickstart Guide
=======================


This guide is written for those who just can't wait to get their hands dirty and will touch on the most common concepts you will encounter as you build your own
projects with Isaac Lab! This includes installation, running RL, finding environments, creating new projects, and more!

The power of Isaac Lab comes from from a few key features that we will very briefly touch on in this guide.

1) **Vectorization**: Reinforcement Learning requires attempting a task many times. Isaac Lab speeds this process along by vectorizing the
   environment, a process by which training can be run in parallel across many copies of the same environment, thus reducing the amount of time
   spent on collecting data before the weights of the model can be updated. Most of the codebase is devoted to defining those parts of the environment
   that need to be touched by this vectorization system

2) **Modular Design**: Isaac Lab is designed to be modular, meaning that you can design your projects to have various components that can be
   swapped out for different needs. For example, suppose you want to train a policy that supports a specific subset of robots.  You could design
   the environment and task to be robot agnostic by writing a controller interface layer in the form of one of our Manager classes (the ``ActionManager``
   in this specific case). Most of the rest of the codebase is devoted to defining those parts of your project that need to be touched by this manager system.

To get started, we will first install Isaac Lab and launch a training script.

Quick Installation Guide
-------------------------

There are many ways to :ref:`install <isaaclab-installation-root>` Isaac Lab, but for the purposes of this quickstart guide, we will follow the
pip install route using virtual environments.

To begin, we first define our virtual environment.


.. code-block:: bash

    # create a virtual environment named env_isaaclab with python3.11
    conda create -n env_isaaclab python=3.11
    # activate the virtual environment
    conda activate env_isaaclab


Next, install a CUDA-enabled PyTorch 2.7.0 build.

   .. code-block:: bash

      pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128


Before we can install Isaac Sim, we need to make sure pip is updated.  To update pip, run

.. tab-set::
    :sync-group: os

    .. tab-item:: :icon:`fa-brands fa-linux` Linux
        :sync: linux

        .. code-block:: bash

            pip install --upgrade pip

    .. tab-item:: :icon:`fa-brands fa-windows` Windows
        :sync: windows

        .. code-block:: batch

            python -m pip install --upgrade pip

and now we can install the Isaac Sim packages.

.. code-block:: none

    pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com

Finally, we can install Isaac Lab.  To start, clone the repository using the following

.. tab-set::

   .. tab-item:: SSH

      .. code:: bash

         git clone git@github.com:isaac-sim/IsaacLab.git

   .. tab-item:: HTTPS

      .. code:: bash

         git clone https://github.com/isaac-sim/IsaacLab.git

Installation is now as easy as navigating to the repo and then calling the root script with the ``--install`` flag!

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         ./isaaclab.sh --install # or "./isaaclab.sh -i"

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: bash

         isaaclab.bat --install :: or "isaaclab.bat -i"


Launch Training
-------------------

The various backends of Isaac Lab are accessed through their corresponding ``train.py`` and ``play.py`` scripts located in the ``isaaclab/scripts/reinforcement_learning`` directory.
Invoking these scripts will require a **Task Name** and a corresponding **Entry Point** to the gymnasium API. For example

.. code-block:: bash

    python scripts/reinforcement_learning/skrl/train.py --task=Isaac-Ant-v0

This will train the mujoco ant to "run".  You can see the various launch option available to you with the ``--help`` flag.  Note specifically the ``--num_envs`` option and the ``--headless`` flag,
both of which can be useful when trying to develop and debug a new environment. Options specified at this level automatically overwrite any configuration equivalent that may be defined in the code
(so long as those definitions are part of a ``@configclass``, see below).

List Available Environments
-----------------------------

Above, ``Isaac-Ant-v0`` is the task name and ``skrl`` is the RL framework being used.  The ``Isaac-Ant-v0`` environment
has been registered with the `Gymnasium API <https://gymnasium.farama.org/>`_, and you can see how the entry point is defined
by calling the ``list_envs.py`` script, which can be found in ``isaaclab/scripts/environments/list_envs.py``. You should see entries like the following

.. code-block:: bash

    $> python scripts/environments/list_envs.py

    +--------------------------------------------------------------------------------------------------------------------------------------------+
    |  Available Environments in Isaac Lab
    +--------+----------------------+--------------------------------------------+---------------------------------------------------------------+
    | S. No. | Task Name            | Entry Point                                | Config
    .
    .
    .
    +--------+----------------------+--------------------------------------------+---------------------------------------------------------------+
    |   2    | Isaac-Ant-Direct-v0  |  isaaclab_tasks.direct.ant.ant_env:AntEnv  |  isaaclab_tasks.direct.ant.ant_env:AntEnvCfg
    +--------+----------------------+--------------------------------------------+---------------------------------------------------------------+
    .
    .
    .
    +--------+----------------------+--------------------------------------------+---------------------------------------------------------------+
    |   48   | Isaac-Ant-v0         | isaaclab.envs:ManagerBasedRLEnv            |   isaaclab_tasks.manager_based.classic.ant.ant_env_cfg:AntEnvCfg
    +--------+----------------------+--------------------------------------------+---------------------------------------------------------------+

Notice that there are two different ``Ant`` tasks, one for a ``Direct`` environment and one for a ``ManagerBased`` environment.
These are the :ref:`two primary workflows<feature-workflows>` that you can use with Isaac Lab out of the box. The Direct workflow will give you the
shortest path to a working custom environment for reinforcement learning, but the Manager based workflow will give your project the modularity required
for more generalized development.  For the purposes of this quickstart guide, we will only focus on the Direct workflow.


Generate Your Own Project
--------------------------

Getting a new project started with Isaac Lab can seem daunting at first, but this is why we provide the :ref:`template
generator<template-generator>`, to rapidly boilerplate a new project via the command line.

.. code-block:: bash

    ./isaaclab.sh --new

This will create a new project for you based on the settings you choose

* **External vs Internal**: Determines if the project is meant to be built as a part of the isaac lab repository, or if
  it is meant to be loaded as an external extension.
* **Direct vs Manager**: A direct task primarily contains all the implementation details within the environment definition,
  while a manager based project is meant to use our modular definitions for the different "parts" of an environment.
* **Framework**: You can select more than one option here.  This determines which RL frameworks you intend to natively use with your project
  (which specific algorithm implementations you want to use for training).

Once created, navigate to the installed project and run

.. code-block:: bash

    python -m pip install -e source/<given-project-name>

to complete the installation process and register the environment.  Within the directories created by the template
generator, you will find at least one ``__init__.py`` file with something that looks like the following

.. code-block:: python

    import gymnasium as gym

    gym.register(
        id="Template-isaaclabtutorial_env-v0",
        entry_point=f"{__name__}.isaaclabtutorial_env:IsaaclabtutorialEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.isaaclabtutorial_env_cfg:IsaaclabtutorialEnvCfg",
            "skrl_cfg_entry_point": f"{agents.__name__}.skrl_ppo_cfg:PPORunnerCfg",
        },
    )

This is the function that actually registers an environment for future use.  Notice that the ``entry_point`` is literally
just the python module path to the environment definition.  This is why we need to install the project as a package: the module path **is** the
entry point for the gymnasium API.

Configurations
---------------

Regardless of what you are going to be doing with Isaac Lab, you will need to deal with **Configurations**. Configurations
can all be identified by the inclusion of the ``@configclass`` decorator above their class definition and the lack of an ``__init__`` function. For example, consider
this configuration class for the :ref:`cartpole environment <tutorial-create-direct-rl-env>`.

.. code-block:: python

    @configclass
    class CartpoleEnvCfg(DirectRLEnvCfg):
        # env
        decimation = 2
        episode_length_s = 5.0
        action_scale = 100.0  # [N]
        action_space = 1
        observation_space = 4
        state_space = 0

        # simulation
        sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

        # robot
        robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        cart_dof_name = "slider_to_cart"
        pole_dof_name = "cart_to_pole"

        # scene
        scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

        # reset
        max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
        initial_pole_angle_range = [-0.25, 0.25]  # the range in which the pole angle is sampled from on reset [rad]

        # reward scales
        rew_scale_alive = 1.0
        rew_scale_terminated = -2.0
        rew_scale_pole_pos = -1.0
        rew_scale_cart_vel = -0.01
        rew_scale_pole_vel = -0.005

Notice that the entire class definition is just a list of value fields and other configurations. Configuration classes are
necessary for anything that needs to care about being vectorized by the lab during training. If you want to be able to copy an
environment thousands of times, and manage the data from each asynchronously, you need to somehow "label" what parts of the scene matter
to this copying process (vectorization). This is what the configuration classes accomplish!

In this case, the class defines the configuration for the entire training environment! Notice also the ``num_envs`` variable in the ``InteractiveSceneCfg``. This actually gets overwritten
by the CLI argument from within the ``train.py`` script.  Configurations provide a direct path to any variable in the configuration hierarchy, making it easy
to modify anything "configured" by the environment at launch time.

Robots
-------

Robots are entirely defined as instances of configurations within Isaac Lab.  If you examine ``source/isaaclab_assets/isaaclab_assets/robots``, you will see a number of files, each of which
contains configurations for the robot in question.  The purpose of these individual files is to better define scope for all the different robots, but there is nothing preventing
you from :ref:`adding your own <tutorial-add-new-robot>` to your project or even to the ``isaaclab`` repository! For example, consider the following configuration for
the Dofbot

.. code-block:: python

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets.articulation import ArticulationCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    DOFBOT_CONFIG = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Dofbot/dofbot.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint1": 0.0,
                "joint2": 0.0,
                "joint3": 0.0,
                "joint4": 0.0,
            },
            pos=(0.25, -0.25, 0.0),
        ),
        actuators={
            "front_joints": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-2]"],
                effort_limit_sim=100.0,
                velocity_limit_sim=100.0,
                stiffness=10000.0,
                damping=100.0,
            ),
            "joint3_act": ImplicitActuatorCfg(
                joint_names_expr=["joint3"],
                effort_limit_sim=100.0,
                velocity_limit_sim=100.0,
                stiffness=10000.0,
                damping=100.0,
            ),
            "joint4_act": ImplicitActuatorCfg(
                joint_names_expr=["joint4"],
                effort_limit_sim=100.0,
                velocity_limit_sim=100.0,
                stiffness=10000.0,
                damping=100.0,
            ),
        },
    )

This completely defines the dofbot! You could copy this into a ``.py`` file and import it as a module and you would be able to use the dofbot in
your own lab sims. One common feature you will see in any config defining things with state is the presence of an ``InitialStateCfg``.  Remember, the configurations
are what informs vectorization, and it's the ``InitialStateCfg`` that describes the state of the joints of our robot when it gets created in each environment. The
``ImplicitActuatorCfg`` defines the joints of the robot using the default actuation model determined by the joint time.  Not all joints need to be actuated, but you
will get warnings if you don't.  If you aren't planning on using those undefined joints, you can generally ignore these.

Apps and Sims
--------------

Using the simulation means launching the Isaac Sim app to provide simulation context. If you are not running a task defined by the standard workflows, then you
are responsible for creating the app, managing the context, and stepping the simulation forward through time.  This is the "third workflow": a **Standalone** app, which
is what we call the scripts for the frameworks, demos, benchmarks, etc...

The Standalone workflow gives you total control over *everything* in the app and simulation
context. Developing standalone apps is discussed at length in the `Isaac Sim documentation <https://docs.isaacsim.omniverse.nvidia.com/latest/index.html>`_ but there
are a few points worth touching on that can be incredibly useful.

.. code-block:: python

    import argparse

    from isaaclab.app import AppLauncher
    # add argparse arguments
    parser = argparse.ArgumentParser(
        description="This script demonstrates adding a custom robot to an Isaac Lab environment."
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    # parse the arguments
    args_cli = parser.parse_args()

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

The ``AppLauncher`` is the entrypoint to any and all Isaac Sim applications, like Isaac Lab! *Many Isaac Lab and Isaac Sim modules
cannot be imported until the app is launched!*.  This is done on the second to last line of the code above, when the ``AppLauncher`` is constructed.
The ``app_launcher.app`` is our interface to the Kit App Framework; the broader interstitial code that binds the simulation to things the extension
management system, or the GUI, etc...  In the standalone workflow, this interface, often called the ``simulation_app`` is predominantly used
to check if the simulation is running, and cleanup after the simulation finishes.
