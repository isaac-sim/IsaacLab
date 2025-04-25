.. _walkthrough_designing_the_env:

Designing the Environment
==========================

Now that we have our project installed, we can start designing the environment. In the traditional description 
of a reinforcement learning problem, the environment is responsible for using the actions produced by the agent to 
update the the state of the "world", and finally compute and return the observations and the reward signal.

Our template is set up for the **direct** workflow, which means the environment class will manage all of these details 
centrally. We will need to write the code that will...

1. Define the training simulation and manage cloning
2. Define the robot
3. Apply the actions from the agent to the robot
4. Calculate and return the rewards and observations
5. Manage resetting and terminal states

Class and Config
-----------------

To begin, navigate to the task: ``source/isaac_lab_tutorial/isaac_lab_tutorial/tasks/direct/isaac_lab_tutorial`` , and take a look
and the contents of ``isaac_lab_tutorial_env_cfg.py``.  You should see something that looks like the following 

.. code-block:: python

  from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

  from isaaclab.assets import ArticulationCfg
  from isaaclab.envs import DirectRLEnvCfg
  from isaaclab.scene import InteractiveSceneCfg
  from isaaclab.sim import SimulationCfg
  from isaaclab.utils import configclass

  
  @configclass
  class IsaacLabTutorialEnvCfg(DirectRLEnvCfg):

      # Some useful fields 
      .
      .
      .

      # simulation
      sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=2)

      # robot(s)
      robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

      # scene
      scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

      # Some more useful fields 
      .
      .
      .

This is the default configuration for a simple cartpole environment that comes with the template and defines the ``self`` scope 
for anything you do within the corresponding environment.

.. currentmodule:: isaaclab.envs

The first thing to note is the presence of the ``@configclass`` decorator. This defines a class as a configuration class, which hold
a special places in Isaac Lab. Configuration classes are part of how Isaac Lab determines what to "care" about when it comes to cloning 
the environment to scale up training. Isaac Lab provides different base configuration classes depending on your goals, and in this
case we are using the :class:`DirectRLEnvCfg` class because we are interested in performing reinforcement learning in the direct workflow. 

.. currentmodule:: isaaclab.sim

The second thing to note is the content of of the configuration class. As the author, you can specify any fields you desire but, generally speaking, there are three things you 
will always define here: The **sim**, the **scene**, and the **robot**. Notice that these fields are also configuration classes! Configuration classes 
are compositional in this way as a solution for cloning arbitrarily complex environments.

The **sim** is an instance of :class:`SimulationCfg`, and this is the config that controls the nature of the simulated reality we are building. This field is a member 
of the base class, ``DirecRLEnvCfg``, but has a default sim configuration, so it's *technically* optional.   The ``SimulationCfg`` dictates 
how finely to step through time (dt), the direction of gravity, and even how physics should be simulated. In this case we only specify the time step and the render interval, with the 
former indicating that each step through time should simulate :math:`1/120`th of a second, and the latter being how many steps we should take before we render a frame (a value of 2 means 
render every other frame).

.. currentmodule:: isaaclab.scene

the **scene** is an instance of :class:`InteractiveSceneCfg`. The scene describes what goes "on the stage" and manages those simulation entities to be cloned across environments.
The scene is also a member of the base class ``DirectRLEnvCfg``