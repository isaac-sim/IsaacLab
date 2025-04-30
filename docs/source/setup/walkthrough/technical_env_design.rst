.. _walkthrough_technical_env_design:

Environment Design
====================

Armed with our understanding of the project and its structure, we are ready to start modifying the code to suit our Jetbot training needs.
Our template is set up for the **direct** workflow, which means the environment class will manage all of these details 
centrally. We will need to write the code that will...

#. Define the robot
#. Define the training simulation and manage cloning
#. Apply the actions from the agent to the robot
#. Calculate and return the rewards and observations
#. Manage resetting and terminal states

As a first step, our goal will be to get the environment training pipeline to load and run.  We will use a dummy reward signal 
for the purposes of this part of the walkthrough.

Define the Robot
------------------

As our project grows, we may have many robots that we want to train. With malice aforethought we will add a new ``module`` to our 
tutorial ``extension`` named ``robots`` where we will keep the definitions for robots as individual python scripts. Navigate 
to ``isaac_lab_tutorial/source/isaac_lab_tutorial/isaac_lab_tutorial`` and create a new folder called ``robots``. Within this folder
create two files: ``__init__.py`` and ``jetbot.py``. The ``__init__.py`` file marks this directory as a python module and we will 
be able to import the contents of ``jetbot.py`` in the usual way.

The contents of ``jetbot.py`` is fairly minimal

.. code-block:: python

  import isaaclab.sim as sim_utils
  from isaaclab.assets import ArticulationCfg
  from isaaclab.actuators import ImplicitActuatorCfg
  from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

  JETBOT_CONFIG = ArticulationCfg(
      spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Jetbot/jetbot.usd"),
      actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
  )

The only purpose of this file is to define a unique scope in which to save our configurations. The details of robot configurations 
can be explored in :ref:`this tutorial <tutorial-add-new-robot>` but most noteworthy for this walkthrough is the ``usd_path`` for the ``spawn``
argument of this ``ArticulationCfg``. The Jetbot asset is available to the public via a hosted nucleus server, and that path is defined by 
``ISAAC_NUCLEUS_DIR``, however any path to a USD file is valid, including local ones! 

Environment Configuration 
---------------------------



.. code-block:: python

  from isaac_lab_tutorial.robots.jetbot import JETBOT_CONFIG

  from isaaclab.assets import ArticulationCfg
  from isaaclab.envs import DirectRLEnvCfg
  from isaaclab.scene import InteractiveSceneCfg
  from isaaclab.sim import SimulationCfg
  from isaaclab.utils import configclass

  @configclass
  class IsaacLabTutorialEnvCfg(DirectRLEnvCfg):
      # env
      decimation = 2
      episode_length_s = 5.0
      # - spaces definition
      action_space = 2
      observation_space = 6
      state_space = 0

      # simulation
      sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

      # robot(s)
      robot_cfg: ArticulationCfg = JETBOT_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")

      # scene
      scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=4.0, replicate_physics=True)

      dof_names = ["left_wheel_joint", "right_wheel_joint"]