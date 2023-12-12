Creating an RL Environment
==========================

In Orbit, we provide a set of environments that are ready to use. However, you may want to create your own
RL environment for your application. This tutorial will show you how to create a new RL environment from scratch.

As a practice, we maintain all the environments that are *officially* provided in the ``omni.isaac.orbit_tasks``
extension. It is recommended to add your environment to the extension ``omni.isaac.contrib_tasks``. This way, you can
easily update your environment when the API changes and you can also contribute your environment to the community.

In this tutorial, we will look at the configuration class :class:`RLTaskEnvCfg` that is used to
configure your learning agent and discuss the different classes you need to create to configure
your RL task. We will use the Cartpole balancing task environment as an example to illustrate the
different components.

The Code
~~~~~~~~

All RL environments in Orbit inherit from the base class :class:`RLTaskEnv`. The base class follows the ``gym.Env``
interface and provides the basic functionality for an environment. Similar to `IsaacGym <https://sites.google.com/view/isaacgym-nvidia>`_,
all environments designed in Orbit are *vectorized* implementations. This means that multiple environment
instances are packed into the simulation and the user can interact with the environment by passing in a batch of actions.

.. note::

   While the environment itself is implemented as a vectorized environment, we do not
   inherit from :class:`gym.vector.VectorEnv`. This is mainly because the class adds
   various methods (for wait and asynchronous updates) which are not required.
   Additionally, each RL library typically has its own definition for a vectorized
   environment. Thus, to reduce complexity, we directly use the :class:`gym.Env` over
   here and leave it up to library-defined wrappers to take care of wrapping this
   environment for their agents.

The base class :class:`RLTaskEnv` wraps around many intricacies of the simulation and
provides a simple interface for the user to implement their own environment. At the core, the
base class provides the following functionality:

* :meth:`__init__` method to create the environment instances and initialize different components
* :meth:`reset` and :meth:`step` methods that are used to interact with the environment
* :meth:`render` method to render the environment
* :meth:`close` method to close the environment

All environments are registered using the :func:`gym.register` method. This method takes in the name of the
environment, the class that implements the environment and the configuration file for the environment.
The name of the environment is used to create the environment using the :func:`gym.make` method.

The base class :class:`RLTaskEnv` is defined in the file ``rl_task_env.py``:

.. dropdown:: :fa:`eye,mr-1` Code for ``rl_task_env.py``

   .. literalinclude:: ../../../../source/extensions/omni.isaac.orbit/omni/isaac/orbit/envs/rl_task_env.py
      :language: python
      :linenos:

Similar to other components of Orbit, instead of directly modifying the base
class :class:`RLTaskEnv`, users can simply implement a configuration
:class:`RLTaskEnvCfg` which will then be used to construct a
:class:`RLTaskEnv` instance.

This tutorial will continue along with the Cartpole example, this time creating a `RLTaskEnvCfg`
to define the task of balancing the pole from a reinforcement learning perspective.

.. dropdown:: :fa:`eye,mr-1` Code for ``cartpole_env_cfg.py``

   .. literalinclude:: ../../../../source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/classic/cartpole/cartpole_env_cfg.py
      :language: python
      :linenos:



The Code Explained
~~~~~~~~~~~~~~~~~~

Designing the scene
-------------------

The first step in creating a new environment is to design the scene in which the agent will operate within.
The scene used in this tutorial is the same one used in the tutorial :ref:`creating-base-env`, so we won't
go over it in detail again here.

Also see :ref:`tutorial-interactive-scene` for even more details on scene creation.

Designing the Action and Observation Spaces
-------------------------------------------

Again, the :class:`ActionTerm` and :class:`ObservationTerm` used here are the same as those
used in :ref:`creating-base-env`, so you can reference that tutorial for more details.


Designing the Rewards
------------------------

The :class:`RewardsCfg` configures the :class:`RewardManager` to dictate how the agent receives rewards from the
environment. In this example we define a few reward terms to guide our agent to a robust policy to balance the pole.

To define a reward term, you need to provide the function that computes the reward
as well as the weighting associated with it.

There are a few reward functions pre-defined in ``source/extensions/omni.isaac.orbit/omni/isaac/orbit/envs/mdp/rewards.py``
that will be used in the Cartpole environment, but when creating your own environments, feel free to add more to
your task config as you see fit.

The various reward terms used in this environment will be explained in the following sections. Feel free to skip over this
if you are already familiar with the cartpole example.

.. literalinclude:: ../../../../source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/classic/cartpole/cartpole_env_cfg.py
   :language: python
   :start-after: # Rewards configuration
   :end-before: # Terminations configuration

* **Alive Reward Term**: Agent receives a reward at each step that it is
  not terminated which is weighted by a factor of 1.0. This term is used to encourage the agent to
  stay alive for as long as possible.
* **Terminating Reward Term**: Agent receives a reward at each step that it is in terminated state
  which is weighted by a factor of -2.0. This term is similarly used to penalize the agent for terminating.
* **Pole Angle Reward Term**: Agent receives a reward based upon the L2 norm of the current pole angle
  compared to the target pole angle which is weighted by a factor of -1.0. This term is used to encourage
  the agent to keep the pole angle close to the desired angle.
* **Cart Velocity Reward Term**: Agent receives a reward based upon the L1 norm of the current cart velocity
  which is weighted by a factor of -0.01. This term is used to encourage the agent to keep the cart velocity
  close to the desired velocity.
* **Pole Velocity Reward Term**: Agent receives a reward based upon the L1 norm of the current pole velocity
  which is weighted by a factor of -0.005. This term is used to encourage the agent to keep the pole velocity
  close to the desired velocity.

Designing the Termination Criteria
----------------------------------

In RL tasks, it is important to define when an episode is terminated. This is because the agent needs to know when
to reset the environment and start a new episode.

The :class:`TerminationsCfg` configures what constitutes an episode as
terminated. In this example, we want the task to terminate when either of the following conditions is met:
* **Episode Length** The episode length is greater than the defined max_episode_length
* **Cart out of bounds** The cart goes outside of the bounds [-3, 3]

.. literalinclude:: ../../../../source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/classic/cartpole/cartpole_env_cfg.py
   :language: python
   :start-after: # Terminations configuration
   :end-before: # Curriculum configuration


As with :class:`RewardTermCfg`, you can define additional :class:`DoneTermCfg` for any additional criteria
for which you want to terminate the episode and add them to the :class:`TerminationsCfg`.

Curriculum and Commands
-----------------------

Curriculum
^^^^^^^^^^

Often times when training an agent, it is useful to start with a simple task and gradually increase the difficulty
as the agent learns. This is the idea behind curriculum learning. In Orbit, we provide a :class:`CurriculumManager`
that can be used to define a curriculum for your environment. In this tutorial we won't implement one for simplicity,
but you can see an example of a curriculum definition in the included Lift task for inspiration.

.. TODO: add tutorial that explains how to use the curriculum manager and reference it here.

We use a simple pass-through curriculum in this example to define a curriculum manager that does not modify the
environment.

.. literalinclude:: ../../../../source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/classic/cartpole/cartpole_env_cfg.py
   :language: python
   :start-after: # Curriculum configuration
   :end-before: ##

Commands
^^^^^^^^

Additionally, you can also define commands that are sent to the environment at the start of each episode. This is
useful for resetting the environment to a specific state or for providing additional information to the agent. In this
example, we don't use any commands, but you can see an example of a command definition in the included Lift task for
inspiration.

.. TODO: add tutorial that explains how to use the command manager and reference it here.

We use the :class:`NullCommandGeneratorCfg` in this example to define a command generator that does not generate
any commands. It is provided to the user as a convenience class to avoid having to define a new command generator.
There are other command generators that are provided in
``source/extensions/omni.isaac.orbit/omni/isaac/orbit/command_generators``.

.. dropdown:: :fa:`eye,mr-1` Code for ``null_command_generator.py``

   .. literalinclude:: ../../../../source/extensions/omni.isaac.orbit/omni/isaac/orbit/command_generators/null_command_generator.py
      :language: python
      :linenos:

Combining Everything
--------------------

Now we want to construct a :class:`RLTaskEnvCfg` that ties all of the
task configurations together. This object will be used to construct the
:class:`RLEnv` to be used for RL training.

Again this is similar to the :class:`BaseEnvCfg` defined in :ref:`creating-base-env`.
only with the added RL components explained in the above sections:
* Curriculum
* Rewards
* Terminations
* Commands

.. TODO: Explain __post_init__

Registering the environment
---------------------------

Before you can run your environment, you need to register your environment with the OpenAI Gym interface.

To register an environment, call the :meth:`gym.register` method in the :mod:`__init__.py` file of your environment package
(for instance, in ``omni.isaac.contrib_tasks.__init__.py``). This has the following components:

* **Name of the environment:** This should ideally be in the format :const:`Isaac-\<EnvironmentName\>-\<Robot\>-\<Version\>`.
  However, this is not a strict requirement and you can use any name you want.
* **Entry point:** This is the import path of the environment class. This is used to instantiate the environment.
* **Config entry point:** This is the import path of the environment configuration file. This is used to instantiate the environment configuration.
  The configuration file can be either a YAML file or a Python dataclass

As examples of this in the ``omni.isaac.orbit_tasks`` package, we have the following:

.. dropdown:: :fa:`eye,mr-1` Registering an environment with a YAML configuration file

   .. literalinclude:: ../../../../source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/__init__.py
      :language: python
      :lines: 52-56
      :linenos:
      :lineno-start: 52

.. dropdown:: :fa:`eye,mr-1` Registering an environment with a Python dataclass configuration file

   .. literalinclude:: ../../../../source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/__init__.py
      :language: python
      :lines: 84-88
      :linenos:
      :lineno-start: 84


The Code Execution
~~~~~~~~~~~~~~~~~~

Now that we have gone through the code, let's run the environment. All environments registered in the ``omni.isaac.orbit_tasks``
and ``omni.isaac.contrib_tasks`` packages are automatically available in the included standalone environments and workflows scripts.

As an example, to run the Cartpole RL environment, you can use the following command.

.. code-block:: bash

   ./orbit.sh -p source/standalone/environments/random_agent.py --task Isaac-Cartpole-v0 --num_envs 32


This should open a stage with a ground plane, lights, and 4096 Cartpole agents initialized at different
random configurations. The simulation should be playing with each Carpoles moving randomly.
To stop the simulation, you can either close the window, or press the ``STOP`` button in the UI, or press ``Ctrl+C`` in the terminal
where you started the simulation.
