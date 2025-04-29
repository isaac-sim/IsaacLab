.. _walkthrough_concepts_env_design:

Environment Design Background
==============================

Now that we have our project installed, we can start designing the environment. In the traditional description 
of a reinforcement learning (RL) problem, the environment is responsible for using the actions produced by the agent to 
update the the state of the "world", and finally compute and return the observations and the reward signal. However, there are 
some additional concepts that are unique to Isaac Sim and Lab regarding the mechanics of the simulation itself.  
The traditional description of a reinforcement learning problem presumes a "world", but we get no such luxury; we must define 
the world ourselves, and success depends on understanding on how to construct that world and how it will fit into the simulation

App, Sim, World, Stage, and Scene
----------------------------------

.. figure:: ../../_static/setup/walkthrough_sim_stage_scene.svg
    :align: center
    :figwidth: 100%
    :alt: How the sim is organized.

The **World** is defined by the origin of a cartesian coordinate system and the units that define it. How big or how small? How
near or how far?  The answers to questions like these can only be defined *relative* to some contextual reference frame, and that 
reference frame is what defines the world.

"Above" the world in structure is the **Sim**\ ulation and the **App**\ licaiton.  The **Application** is "the thing responsible for 
everything else": It governs all resource management as well as launching and destroying the simulation when we are done with it.  
When we :ref:`launched training with the template<walkthrough_project_setup>`, the window that appears with the viewport of cartpoles 
training is the Application window.  The application is not defined by the GUI however, and even when running in headless mode all 
simulations have an application that governs them.

The **Simulation** controls the "rules" of the world.  It defines the laws of physics, such as how time and gravity should work, and how frequently to perform 
rendering. If the application holds the sim, then the sim holds the world. The simulation governs a single step through time by dividing it into many different
sub-steps, each devoted to a specific aspect of updating the world into a the state. Many of the APIs in Isaac Lab are written to specifically hook into
these various steps and you will often see functions named things like ``_pre_XYZ_step`` and ``_post_XYZ_step`` where ``XYZ_step`` is the name of one of these sub-steps of 
the simulation, such as the ``physics_step`` or the ``render_step``.

"Below" the world in structure is the **Stage** and the **Scene**.  If the world provides spatial context to the sim, then 
the **Stage** provides the *compositional context* for the world. Suppose we want to simulate a table set for a meal in a room: 
the room is the "world" in this case, and we choose the origin of the world to be one of the corners of the room. The position of the 
table in the room is defined as a vector from the origin of the world to some point on the table that we choose to be the origin of a *new* coordinate 
system, fixed to the table.  It's not useful to us, *the agent*\ , to talk about the location of the food and the utensils on the table with respect to the 
corner of the room: instead it is preferable to use the coordinates defined with respect to the table. However, the simulation needs to know 
these global coordinates in order to properly simulate the next time step, so we must define how these two coordinate systems are *composed* together.

This is what the stage accomplishes: everything in the simulation is a `USD primitive <https://openusd.org/release/glossary.html#usdglossary-prim>`_ and the 
stage represents the relationships between these primitives as a tree, with the context being defined by the relative path in the tree. Every prim on the stage 
has a name and therefore a path in this tree, such as ``/room/table/food`` or ``room/table/utensils``. Relationships are defined by the "parents" and "children" 
of a given node in this tree: the ``table`` is a child of the ``room`` but a parent of ``food``. Compositional properties of the parent are applied to all of its 
children, but child prims have the ability to override parent properties if necessary, as is often the case for materials.

.. figure:: ../../_static/setup/walkthrough_stage_context.svg
    :align: center
    :figwidth: 100%
    :alt: How the stage organizes context

Armed with this vocabulary, we can finally talk about the **Scene**, one of the most critical elements to understand about Isaac Lab. Deep learning, in 
all its forms, is rooted in the analysis of data.  This is true even in robot learning, where data is acquired through the sensors of the robot being trained.
The time required to setup the robot, collect data, and reset the robot to collect more, is a fundamental bottleneck in teaching robots to do *anything*, with any method.
Isaac Sim gives us access to robots without the need for literal physical robots, but Isaac Lab gives us access to *vectorization*: the ability to simulate many copies 
of a training procedure efficiently, thus multiplying the rate of data generation and accelerating training proportionally. The scene governs those primitives on the stage 
that matter to this vectorization process, known as **simulation entities**.

Suppose the reason why you want to simulate a table set for a meal is because you would like to train a robot to place the table settings for you! The robot, the table, 
and all the things on it can be registered to the scene of an environment.  We can then specify how many copies we want and the scene will automatically 
construct and run those copies on the stage. These copies are placed at new coordinates on the stage, defining a new reference frame from which observations 
and rewards can be computed. Every copy of the scene exists on the stage and is being simulated by the same world.  This is much more efficient 
than running unique simulations for each copy, but it does open up the possibility of unwanted interactions between copies of the scene, so it's important 
to keep this in mind while debugging.

Now that we have a grasp on the mechanics, we can take a look at the code generated for our template project!

Classes and Configs
---------------------

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
The scene is also a member of the base class ``DirectRLEnvCfg``, but unlike the sim it has no default an must be defined in every ``DirectRLEnvCfg``.  The ``InteractiveSceneCfg``
describes how many copies of the scene we want to create for training purposes, as well as how far apart they should be spaced on the stage.

.. currentmodule:: isaaclab.assets

Finally we have the **robot** definition, which is an instance of  :class:`ArticulationCfg`. An environment could have multiple articulations, and so the presence of
an ``ArticulationCfg`` is not strictly required in order to define a ``DirectRLEnv``.  Instead, the usual workflow is to define a regex path to the robot, and replace 
the ``prim_path`` attribute in the base configuration. In this case, ``CARTPOLE_CFG`` is a configuration defined in ``isaaclab_assets.robots.cartpole`` and by replacing 
the prim path with ``/World/envs/env_.*/Robot`` we are implicitly saying that every copy of the scene will have a robot named ``Robot``. 


The Environment
-----------------

Next, let's take a look at the contents the other python file in our task directory: ``isaac_lab_tutorial_env_cfg.py``

.. code-block:: python

  #imports 
  .
  .
  .
  from .isaac_lab_tutorial_env_cfg import IsaacLabTutorialEnvCfg

  class IsaacLabTutorialEnv(DirectRLEnv):
      cfg: IsaacLabTutorialEnvCfg

      def __init__(self, cfg: IsaacLabTutorialEnvCfg, render_mode: str | None = None, **kwargs):
          super().__init__(cfg, render_mode, **kwargs)
          . . .

      def _setup_scene(self):
          self.robot = Articulation(self.cfg.robot_cfg)
          # add ground plane
          spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
          # add articulation to scene
          self.scene.articulations["robot"] = self.robot
          # clone and replicate
          self.scene.clone_environments(copy_from_source=False)
          # add lights
          light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
          light_cfg.func("/World/Light", light_cfg)

      def _pre_physics_step(self, actions: torch.Tensor) -> None:
          . . .

      def _apply_action(self) -> None:
          . . .

      def _get_observations(self) -> dict:
          . . .

      def _get_rewards(self) -> torch.Tensor:
          total_reward = compute_rewards(...)
          return total_reward

      def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
          . . .

      def _reset_idx(self, env_ids: Sequence[int] | None):
          . . .

  @torch.jit.script
  def compute_rewards(...):
      . . .
      return total_reward


.. currentmodule:: isaaclab.envs

Some of the code has been omitted for clarity, in order to aid in discussion. This is where the actual "meat" of the 
direct workflow exists and where most of our modifications will take place as we tweak the template to suit our needs.
Currently, all of the member functions of ``IsaacLabTutorialEnv`` are directly inherited from the :class:`DirectRLEnv`. This 
known interface is how Isaac Lab and its supported RL frameworks interact with the environment. 

When the environment is initialized it receives its own config as an argument, which is then immediately passed to super in order
to initialize the ``DirectRLEnv``.  This super call also calls ``_setup_scene``, which actually constructs the scene and and clones 
it appropriately. Notably is how the robot is create and registered to the scene in ``_setup_scene``.  First, the robot articulation 
is created by using the ``robot_config`` we defined in ``IsaacLabTutorialEnvCfg``: it doesn't exist before this point! When the 
articulation is created, the robot exists on the stage at ``/World/envs/env_0/Robot``.  The call to ``scene.clone_environments`` then 
copies ``env_0`` appropriately.  At this point the robot exists as many copies on the stage, so all that's left is to notify the ``scene``
object of the existence of this articulation to be tracked.  The articulations of the scene are kept as a dictionary, so ``scene.articulations["robot"] = self.robot``
creates a new ``robot`` element of the ``articulations``dictionary and sets the value to be ``self.robot``.

Notice also that the remaining functions do not take additional arguments save ``_reset_idx``.  This is because the environment only manages the application of 
actions to the agent being simulated, and then updating the sim.  This is what the ``_pre_physics_step`` and ``_apply_action`` steps are for: we set the drive commands 
to the robot so that when the simulation steps forward, the actions are applied and the joints are driven to new targets. This process is broken into steps like this 
in order to ensure systematic control over how the environment is executed, and is especially important in the manager workflow. A similar relationship exists between the 
``_get_dones`` function and ``_reset_idx``.  The former, ``_get_dones`` determines if each of the environments is in a terminal state, and populates tensors of boolean 
values to indicate which environments terminated due to entering a terminal state vs time out (the two returned tensors of the function).  The latter, ``_reset_idx`` takes a 
list environment index values (integers) and then actually resets those environments.  It is important that things like updating drive targets or resetting environments 
do not happen **during** the physics or rendering steps, and breaking up the interface in this way helps prevent that.
