Creating an environment
=======================

In Orbit, we provide a set of environments that are ready to use. However, you may want to create your own
environment for your application. This tutorial will show you how to create a new environment from scratch.

As a practice, we maintain all the environments that are *officially* provided in the ``omni.isaac.orbit_envs``
extension. It is recommended to add your environment to the extension ``omni.isaac.contrib_envs``. This way, you can
easily update your environment when the API changes and you can also contribute your environment to the community.

In this tutorial, we will look at the base class :py:class:`IsaacEnv` and discuss the different methods that you
need to implement to create a new environment. We will use the Lift environment as an example to illustrate the
different steps.

The Code
~~~~~~~~

All environments in Orbit inherit from the base class :py:class:`IsaacEnv`. The base class follows the ``gym.Env``
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

The base class :py:class:`IsaacEnv` wraps around many intricacies of the simulation and
provides a simple interface for the user to implement their own environment. At the core, the
base class provides the following functionality:

* :py:meth:`__init__` method to create the environment instances and initialize different components
* :py:meth:`reset` and :py:meth:`step` methods that are used to interact with the environment
* :py:meth:`render` method to render the environment
* :py:meth:`close` method to close the environment

The user needs to implement the following methods for their environment:

* :py:meth:`_design_scene` method to design the scene for a single environment instance
* :py:meth:`_reset_idx` method to reset a environment instances based on the provided indices
* :py:meth:`_step_impl` method to perform the pre-processing of actions, stepping the simulation and computing reward and termination signals
* :py:meth:`_get_observations` method to compute the observations for the batch of environment instances

Additionally, the following attributes need to be set by the inherited class:

* :py:attr:`action_space`: The Space object corresponding to valid actions. This should correspond to the action space of a single environment instance.
* :py:attr:`observation_space`: The Space object corresponding to valid observations. This should correspond to the observation space of a single environment instance.
* :py:attr:`reward_range`: A tuple corresponding to the min and max possible rewards. A default reward range set to [-inf, +inf] already exists.

All environments are registered using the :py:func:`gym.register` method. This method takes in the name of the
environment, the class that implements the environment and the configuration file for the environment.
The name of the environment is used to create the environment using the :py:func:`gym.make` method.

The base class :py:class:`IsaacEnv` is defined in the file ``isaac_env.py``:

.. dropdown:: :fa:`eye,mr-1` Code for `isaac_env.py`

   .. literalinclude:: ../../../source/extensions/omni.isaac.orbit_envs/omni/isaac/orbit_envs/isaac_env.py
      :language: python
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

Designing the scene
-------------------

The first step in creating a new environment is to design the scene. The scene is designed by the
:py:meth:`_design_scene` method. This method is called once for a single or *template* environment scene
before cloning it for the other environment instances using the `Cloner API <https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_gym_cloner.html>`_
from Isaac Sim.

Essentially, all prims under the prim path ``/World/envs/env_0`` (:py:attr:`IsaacEnv.template_env_ns`)
are cloned for the other environment instances ``/World/envs/env_{N}``. This means that all the prims that you
want to be cloned for the other environment instances should be under this prim path. Additionally, the Cloner API
performs collision filtering of the prims under different environment instances, i.e. the prims under ``/World/envs/env_0``
will not collide with the prims under ``/World/envs/env_1``. If you want to have a prim that is shared
across all the environment instances for collisions (such as the ground plane), the method should return
the list of paths to these prims. As a practice, we recommend that you create the shared prims under the prim
path ``/World`` directly.

An example of the :py:meth:`_design_scene` method is shown below for the Lift environment. The method
creates a robot and a table under the prim path ``/World/envs/env_0``, while creating a shared ground plane
under the prim path ``/World/defaultGroundPlane`` and optionally, markers for debug visualization under the path ``/Visuals``.

.. dropdown:: :fa:`eye,mr-1` Code for `_design_scene` method in `lift_env.py`

   .. literalinclude:: ../../../source/extensions/omni.isaac.orbit_envs/omni/isaac/orbit_envs/manipulation/lift/lift_env.py
      :language: python
      :lines: 75-110
      :linenos:
      :lineno-start: 75

Resetting the environment
-------------------------

The :py:meth:`_reset_idx` method is called to reset the environment instances corresponding to the provided
indices. This method is used at two places:

* When the environment is first created, the :py:meth:`reset` method of the base class is called which
  resets all the environment instances.
* When the environment is stepped, the :py:meth:`step` method of the base class is called which
  resets the environment instances corresponding to the indices that have terminated.

The user needs to implement this method to reset the environment instances corresponding to the provided indices.
This includes resetting the robot and object states, resetting the reward and observation managers, and resetting
various buffers corresponding to episode counter, episode reward, history, sensors etc.

.. dropdown:: :fa:`eye,mr-1` Code for `_reset_idx` method in `lift_env.py`

   .. literalinclude:: ../../../source/extensions/omni.isaac.orbit_envs/omni/isaac/orbit_envs/manipulation/lift/lift_env.py
      :language: python
      :lines: 112-137
      :linenos:
      :lineno-start: 112

An important thing to note about resetting the simulation is that all environment instances are present in the
same simulation stage. This means that you need to make sure that the reset state of prims in different environment
instances are displaced correctly in the simulation stage (i.e. account for the environment instance offset).

An example of this is shown in the :py:meth:`_randomize_object_initial_pose` method for the Lift environment:

.. literalinclude:: ../../../source/extensions/omni.isaac.orbit_envs/omni/isaac/orbit_envs/manipulation/lift/lift_env.py
   :language: python
   :lines: 298-326
   :lineno-start: 298
   :emphasize-lines: 24-25
   :linenos:


Stepping the environment
------------------------

The :py:meth:`_step_impl` method is called to step the environments based on the provided actions. This method
is called by the :py:meth:`step` method of the base class.

The environment specific implementation of this function is responsible for also stepping the simulation. To have
a clean exit when the timeline is stopped through the UI, the implementation should check the simulation status
after stepping the simulator and return if the simulation is stopped.

.. code-block:: python

      # simulate
      self.sim.step(render=self.enable_render)
      # check that simulation is playing
      if self.sim.is_stopped():
         return

In the :py:meth:`_step_impl` method, the user needs to implement the logic to step the environment. This includes
pre-processing the actions (such as scaling, clipping, etc.), computing the commands for the robot using a controller,
stepping the simulator, updating buffers and sensors, and computing the reward and termination conditions. Additionally,
it is important that the user provides the following information based on the type of environment:

* **Time-out condition**: If the environment is based on episodic learning, the user needs to set the ``self.extras["time_outs"]`` buffer
  to ``True`` for the environment indices that have timed out due to episode length. This is sometimes used by RL algorithms to
  compute the reward for the last step of the episode (as highlighted in the `OpenAI Gym documentation <https://github.com/openai/gym/issues/2510>`_).

* **Success condition**: If the environment is goal-conditioned, the user should provide the ``self.extras["is_success"]`` buffer
  to ``True`` for the environment indices that have succeeded in completing the task. This is sometimes useful for goal-conditioned RL algorithms
  or collecting demonstrations.

An example of the :py:meth:`_step_impl` method is shown below for the Lift environment. The method first pre-processes the actions
and computes the robot actions depending on whether a task-space controller or a joint-space controller is used. It then steps
the simulator at a specified , updates the buffers and sensors, and computes the reward and termination conditions.

.. dropdown:: :fa:`eye,mr-1` Code for `_step_impl` method in `lift_env.py`

   .. literalinclude:: ../../../source/extensions/omni.isaac.orbit_envs/omni/isaac/orbit_envs/manipulation/lift/lift_env.py
      :language: python
      :lines: 139-189
      :emphasize-lines: 21-29,42-47
      :linenos:
      :lineno-start: 139

Computing observations
----------------------

The observation is computed by the :py:meth:`_get_observations` method. This method needs to be implemented to computes the observation
for all the environment instances. The returned observations should be a dictionary with the keys corresponding to the group names
and the values corresponding to the observation tensors of shape ``(num_envs, obs_dims)``. This allows the user to define multiple
observation groups that can then be used for different learning paradigms (e.g. for asymmetric actor-critic, one group could be for the RL
policy while the other is for the RL-critic).

While not prescriptive in the base class, it is recommended that the user always define the ``policy`` group which is used as the default
observation group for the environment. This is essential because various wrappers read this group name to unwrap the observations dictionary
for their respective frameworks.

In the lift environment, the observation is computed by the :py:class:`ObservationManager` class. This class is responsible for computing
the observations for the environment by reading data from the various buffers and sensors. More details on the observation manager can be
found in the `MDP managers <../api/orbit.utils.mdp.html>`_ section.

.. dropdown:: :fa:`eye,mr-1` Code for `_get_observations` method in `lift_env.py`

   .. literalinclude:: ../../../source/extensions/omni.isaac.orbit_envs/omni/isaac/orbit_envs/manipulation/lift/lift_env.py
      :language: python
      :lines: 191-193
      :linenos:
      :lineno-start: 191

Registering the environment
---------------------------

Before you can run your environment, you need to register your environment with the OpenAI Gym interface.

To register an environment, call the :py:meth:`gym.register` method in the :py:mod:`__init__.py` file of your environment package
(for instance, in ``omni.isaac.contrib_envs.__init__.py``). This has the following components:

* **Name of the environment:** This should ideally be in the format :py:const:`Isaac-\<EnvironmentName\>-\<Robot\>-\<Version\>`.
  However, this is not a strict requirement and you can use any name you want.
* **Entry point:** This is the import path of the environment class. This is used to instantiate the environment.
* **Config entry point:** This is the import path of the environment configuration file. This is used to instantiate the environment configuration.
  The configuration file can be either a YAML file or a Python dataclass

As examples of this in the ``omni.isaac.orbit_envs`` package, we have the following:

.. dropdown:: :fa:`eye,mr-1` Registering an environment with a YAML configuration file

   .. literalinclude:: ../../../source/extensions/omni.isaac.orbit_envs/omni/isaac/orbit_envs/__init__.py
      :language: python
      :lines: 52-56
      :linenos:
      :lineno-start: 52

.. dropdown:: :fa:`eye,mr-1` Registering an environment with a Python dataclass configuration file

   .. literalinclude:: ../../../source/extensions/omni.isaac.orbit_envs/omni/isaac/orbit_envs/__init__.py
      :language: python
      :lines: 84-88
      :linenos:
      :lineno-start: 84


The Code Execution
~~~~~~~~~~~~~~~~~~

Now that we have gone through the code, let's run the environment. All environments registered in the ``omni.isaac.orbit_envs``
and ``omni.isaac.contrib_envs`` packages are automatically available in the included standalone environments and workflows scripts.

As an example, to run the Lift environment, you can use the following command.

.. code-block:: bash

   ./orbit.sh -p source/standalone/environments/random_agent.py --task Isaac-Lift-Franka-v0 --num_envs 32


This should open a stage with a ground plane, lights, tables, robots, and objects.
The simulation should be playing with the robot arms going to random joint configurations. To stop the simulation,
you can either close the window, or press the ``STOP`` button in the UI, or press ``Ctrl+C`` in the terminal
where you started the simulation.
