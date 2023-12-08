.. _creating-base-env:

Creating a Base Environment
===========================

In Orbit, there are two types of environments: :class:`BaseEnv` and
:class:`RLTaskEnv`. Base environments contain a robot, its action and
observation spaces as well as randomizations (and handling of resets) to be applied to the
environment. Typically, a :class:`BaseEnv` is utilized if one wants
to evaluate an existing control algorithm, mechanical design or do traditional
robot control but doesn't plan on doing RL. This workflow is commonly used in
other simulators such as Gazebo, Mujoco, etc. :class:`BaseEnv` doesn't
contain rewards and terminations, which are common in RL settings. If
interested in doing RL in Orbit, this tutorial is still a good starting point
as :class:`RLTaskEnv` inherits from :class:`BaseEnv` and there's a lot of shared functionality.

In this tutorial, we will look at the base class :class:`BaseEnv` and its
corresponding configuration class :class:`BaseEnvCfg` and
discuss the different configuration classes that need to be implemented to
create a new environment. We will use the Cartpole environment with simple PD
control as an example to illustrate the different steps in developing a
new :class:`BaseEnv`.

The Code
~~~~~~~~

The tutorial corresponds to the ``cartpole_base_env`` script  in the ``orbit/source/standalone/tutorials`` directory.

.. literalinclude:: ../../../../source/standalone/tutorials/04_envs/cartpole_base_env.py
   :language: python

All environments in Orbit inherit from the base class :class:`BaseEnv`.

The base class :class:`BaseEnv` wraps around many intricacies of the
simulation and provides a simple interface for the user to implement their own
environment. At the core, the base class provides the following
functionality:

* :meth:`__init__` method to create the environment instances and initialize
   different components
* :meth:`reset` and :meth:`step` methods that are used to interact with
   the environment
* :meth:`close` method to close the environment
* :meth:`load_managers` method to load the managers that handle actions,
   observations and any randomizations

The base class :class:`BaseEnv` is defined in the file \ ``base_env.py``:

.. dropdown:: :fa:`eye,mr-1` Code for  ``base_env.py``

   .. literalinclude:: ../../../../source/extensions/omni.isaac.orbit/omni/isaac/orbit/envs/base_env.py
      :language: python
      :lines: 45-323

To customize a :class:`BaseEnv` one needs to implement a :class:`BaseEnvCfg`
which configures the action space, observation space and randomizations
associated with the environment. These are utilized by their associated :class:`ManagerBase`
classes to interact with the environment.

The base class :class:`BaseEnvCfg` is defined in the file ``base_env_cfg.py``:

.. dropdown:: :fa:`eye,mr-1` Code for ``base_env_cfg.py``

   .. literalinclude:: ../../../../source/extensions/omni.isaac.orbit/omni/isaac/orbit/envs/base_env_cfg.py
      :language: python
      :lines: 56-91

The Code Explained
~~~~~~~~~~~~~~~~~~

Designing the scene
-------------------

The first step in creating a new environment is to configure the scene by
implementing a :class:`InteractiveSceneCfg`. This will then be used to construct
a :class:`InteractiveScene` which handles spawning of the objects in the scene.

In this tutorial, we will be using the configuration from ``cartpole_scene.py``.
See :ref:`interactive-scene` for a tutorial on how to create it.

The scene used here consists of a ground plane, the cartpole and some lights.

.. dropdown:: :fa:`eye,mr-1` Code for :class:``CartpoleSceneCfg`` class in ``cartpole_scene.py``

   .. literalinclude:: ../../../../source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/classic/cartpole/cartpole_scene.py
      :language: python


Defining actions
----------------

The action space of the Cartpole environment in this example is the force control
over the sliding Cart portion of the cartpole,
moving it horizontally along the rail to balance the pole vertically.

The :class:`ActionTerm` developed in this tutorial implements PD control. In
:meth:`process_actions`, the PD controller calculates the control input
given the current joint position and velocity of the the `cart_to_pole` joint.
This method, in addition to :meth:`apply_actions` are called by the action
manager at each step of the environment to determine the next action and then
apply it.


.. dropdown:: :fa:`eye,mr-1` Code for :class:``ActionsCfg`` class in ``cartpole_base_env.py``

   .. literalinclude:: ../../../../source/standalone/tutorials/04_envs/cartpole_base_env.py
      :language: python
      :start-after: # Cartpole Action Configuration
      :end-before: # Cartpole Observation Configuration

Defining observations
----------------------

The observation space of the environment defines the observed state at
each time step.

The returned observations will be a dictionary with the keys corresponding to the group names
and the values corresponding to the observation tensors of shape ``(num_envs, obs_dims)``.

This allows the user to define multiple
observation groups that can then be used for different learning paradigms (e.g. for asymmetric actor-critic, one group could be for the RL
policy while the other is for the RL-critic).

While not prescriptive in the base class, it is recommended that the user always define the ``policy`` group which is used as the default
observation group for the environment. This is essential because various wrappers read this group name to unwrap the observations dictionary
for their respective frameworks.

In the cartpole environment, the observation is computed by the :class:`ObservationManager` class. This class is responsible for computing
the observations for the environment by reading data from the various buffers and sensors. More details on the observation manager can be
found in the   `MDP managers <../api/orbit/omni.isaac.orbit.managers.html#observation-manager>`_ section.

.. dropdown:: :fa:`eye,mr-1` Code for :class:`ObservationsCfg` class in ``cartpole_base_env.py``

   .. literalinclude:: ../../../../source/standalone/tutorials/04_envs/cartpole_base_env.py
      :language: python
      :start-after: # Cartpole Observation Configuration
      :end-before: # Cartpole Randomization Configuration

Defining randomizations
-----------------------

Often times in robotics, randomness is used to more closely emulate the real world.
In Orbit, :class:`RandomizationManager` is used to manage randomness and define
what environment terms that will be randomized via its :class:`RandomizationCfg`.
In addition, it handles reset calls, so even if you don't want to randomize anything,
you still need to define a :class:`RandomizationCfg` to handle the reset calls.

In this example, the initial slider to cart joint position and velocity are randomized
to be within (-1.0, 1.0) meters and (-0.1, 0.1) radians respectively. Also, the pole joint's position
and velocity are randomized slightly to make the problem more challenging.

When developing your own environments, feel free to add more :class:`RandTerm` as needed or use the
ones pre packaged with Orbit.

Randomization terms have a `mode` associated with them as denoted by the mode argument of
:class:`RandTerm`. The various `mode`s are `"interval", "reset", "startup"`.

Randomization Modes
####################

* `"interval"` `mode` execute randomization at a given fixed interval.
* `"reset"` `mode` execute randomization on every call to an environment's :meth:`reset`.
* `"startup"` `mode` execute randomization only once at environment startup.

In this example, the randomization terms use `reset` mode indicating the
randomization will be applied upon each call to :class:`reset`.

.. dropdown:: :fa:`eye,mr-1` Code for :class:``RandomizationCfg`` class in ``cartpole_base_env.py```

   .. literalinclude:: ../../../../source/standalone/tutorials/04_envs/cartpole_base_env.py
      :language: python
      :start-after: # Cartpole Randomization Configuration
      :end-before: # Cartpole Environment Configuration

Tying it all together
---------------------
In this section we will integrate the scene, observation, action and randomization
configurations built in the previous sections to fully configure the Cartpole
:class:`BaseEnv`.

.. dropdown:: :fa:`eye,mr-1` Code for :class:`CartpoleEnvCfg` class in ``cartpole_base_env.py``

   .. literalinclude:: ../../../../source/standalone/tutorials/04_envs/cartpole_base_env.py
      :language: python
      :start-after: # Cartpole Environment Configuration
      :end-before: # Main

.. note::  To modify any configuration of the :class:`BaseEnvCfg` you can use :class:`__post_init__`
   as is done in this example.


The main method
---------------
Lastly, we define the main method which will handle resetting and stepping of the environment.
At each iteration, we send the target_position - the `action` to the environment and
receive back the observation which is then printed to the console.

.. dropdown:: :fa:`eye,mr-1` Code for :class:`CartpoleEnvCfg` class in ``cartpole_base_env.py``

   .. literalinclude:: ../../../../source/standalone/tutorials/04_envs/cartpole_base_env.py
      :language: python
      :start-after: # Main


The Code Execution
~~~~~~~~~~~~~~~~~~

Now that we have gone through the code, let's run the environment.

As an example, to run the Cartpole base environment script, you can use the following command.

.. code-block:: bash

   ./orbit.sh -p source/standalone/tutorials/04_envs/cartpole_base_env.py


This should open a stage with a ground plane, lights, and a cartpole.
The simulation should be playing with the cartpole attempting to balance itself
such that the pole is vertical. Feel free to modify the P and D gains
to improve the cart's ability to balance the pole vertically.

To stop the simulation,
you can either close the window, or press the ``STOP`` button in the UI, or press ``Ctrl+C`` in the terminal
where you started the simulation.
