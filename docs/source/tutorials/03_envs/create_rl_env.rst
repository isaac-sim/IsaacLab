.. _tutorial-create-rl-env:


Creating an RL Environment
==========================

.. currentmodule:: omni.isaac.orbit

Having learnt how to create a base environment in :ref:`tutorial-create-base-env`, we will now look at how to create a
task environment for reinforcement learning.

The base environment is designed as an sense-act environment where the agent can send commands to the environment
and receive observations from the environment. This minimal interface is sufficient for many applications such as
traditional motion planning and controls. However, many applications require a task-specification which often
serves as the learning objective for the agent. For instance, in a navigation task, the agent may be required to
reach a goal location. To this end, we use the :class:`envs.RLTaskEnv` class which extends the base environment
to include a task specification.

Similar to other components in Orbit, instead of directly modifying the base class :class:`RLTaskEnv`, we
encourage users to  simply implement a configuration :class:`RLTaskEnvCfg` for their task environment.
This practice allows us to separate the task specification from the environment implementation, making it easier
to reuse components of the same environment for different tasks.

In this tutorial, we will configure the cartpole environment using the :class:`RLTaskEnvCfg` to create a task
for balancing the pole upright. We will learn how to specify the task using reward terms, termination criteria,
curriculum and commands.


The Code
~~~~~~~~

For this tutorial, we use the cartpole environment defined in ``omni.isaac.orbit_tasks.classic.cartpole`` module.

.. dropdown:: Code for cartpole_env_cfg.py
   :icon: code

   .. literalinclude:: ../../../../source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/classic/cartpole/cartpole_env_cfg.py
      :language: python
      :emphasize-lines: 63-68, 124-149, 152-162, 165-169, 187-192
      :linenos:

The script for running the environment ``run_cartpole_rl_env.py`` is present in the
``orbit/source/standalone/tutorials/03_envs`` directory. The script is similar to the
``cartpole_base_env.py`` script in the previous tutorial, except that it uses the
:class:`envs.RLTaskEnv` instead of the :class:`envs.BaseEnv`.

.. dropdown:: Code for run_cartpole_rl_env.py
   :icon: code

   .. literalinclude:: ../../../../source/standalone/tutorials/03_envs/run_cartpole_rl_env.py
      :language: python
      :emphasize-lines: 46-50, 64-65
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

We already went through parts of the above in the :ref:`tutorial-create-base-env` tutorial to learn
about how to specify the scene, observations, actions and randomizations. Thus, in this tutorial, we
will focus only on the RL components of the environment.

In Orbit, we provide various implementations of different terms in the :mod:`envs.mdp` module. We will use
some of these terms in this tutorial, but users are free to define their own terms as well. These
are usually placed in their task-specific sub-package
(for instance, in :mod:`omni.isaac.orbit_tasks.classic.cartpole.mdp`).


Defining rewards
----------------

The :class:`managers.RewardManager` is used to compute the reward terms for the agent. Similar to the other
managers, its terms are configured using the :class:`managers.RewardTermCfg` class. The
:class:`managers.RewardTermCfg` class specifies the function or callable class that computes the reward
as well as the weighting associated with it. It also takes in dictionary of arguments, ``"params"``
that are passed to the reward function when it is called.

For the cartpole task, we will use the following reward terms:

* **Alive Reward**: Encourage the agent to stay alive for as long as possible.
* **Terminating Reward**: Similarly penalize the agent for terminating.
* **Pole Angle Reward**: Encourage the agent to keep the pole at the desired upright position.
* **Cart Velocity Reward**: Encourage the agent to keep the cart velocity as small as possible.
* **Pole Velocity Reward**: Encourage the agent to keep the pole velocity as small as possible.

.. literalinclude:: ../../../../source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/classic/cartpole/cartpole_env_cfg.py
   :language: python
   :pyobject: RewardsCfg
   :linenos:
   :lineno-start: 124

Defining termination criteria
-----------------------------

Most learning tasks happen over a finite number of steps that we call an episode. For instance, in the cartpole
task, we want the agent to balance the pole for as long as possible. However, if the agent reaches an unstable
or unsafe state, we want to terminate the episode. On the other hand, if the agent is able to balance the pole
for a long time, we want to terminate the episode and start a new one so that the agent can learn to balance the
pole from a different starting configuration.

The :class:`managers.TerminationsCfg` configures what constitutes for an episode to terminate. In this example,
we want the task to terminate when either of the following conditions is met:

* **Episode Length** The episode length is greater than the defined max_episode_length
* **Cart out of bounds** The cart goes outside of the bounds [-3, 3]

The flag :attr:`managers.TerminationsCfg.time_out` specifies whether the term is a time-out (truncation) term
or terminated term. These are used to indicate the two types of terminations as described in `Gymnasium's documentation
<https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/>`_.

.. literalinclude:: ../../../../source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/classic/cartpole/cartpole_env_cfg.py
   :language: python
   :pyobject: TerminationsCfg
   :linenos:
   :lineno-start: 152

Defining commands
-----------------

For various goal-conditioned tasks, it is useful to specify the goals or commands for the agent. These are
handled through the :class:`managers.CommandManager`. The command manager handles resampling and updating the
commands at each step. It can also be used to provide the commands as an observation to the agent.

For this simple task, we do not use any commands. This is specified by using a command term with the
:class:`envs.mdp.NullCommandCfg` configuration. However, you can see an example of command definitions in the
locomotion or manipulation tasks.

.. literalinclude:: ../../../../source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/classic/cartpole/cartpole_env_cfg.py
   :language: python
   :pyobject: CommandsCfg
   :linenos:
   :lineno-start: 63

Defining curriculum
-------------------

Often times when training a learning agent, it helps to start with a simple task and gradually increase the
tasks's difficulty as the agent training progresses. This is the idea behind curriculum learning. In Orbit,
we provide a :class:`managers.CurriculumManager` class that can be used to define a curriculum for your environment.

In this tutorial we don't implement a curriculum for simplicity, but you can see an example of a
curriculum definition in the other locomotion or manipulation tasks.
We use a simple pass-through curriculum to define a curriculum manager that does not modify the environment.

.. literalinclude:: ../../../../source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/classic/cartpole/cartpole_env_cfg.py
   :language: python
   :pyobject: CurriculumCfg
   :linenos:
   :lineno-start: 165

Tying it all together
---------------------

With all the above components defined, we can now create the :class:`RLTaskEnvCfg` configuration for the
cartpole environment. This is similar to the :class:`BaseEnvCfg` defined in :ref:`tutorial-create-base-env`,
only with the added RL components explained in the above sections.

.. literalinclude:: ../../../../source/extensions/omni.isaac.orbit_tasks/omni/isaac/orbit_tasks/classic/cartpole/cartpole_env_cfg.py
   :language: python
   :pyobject: CartpoleEnvCfg
   :linenos:
   :lineno-start: 177

Running the simulation loop
---------------------------

Coming back to the ``run_cartpole_rl_env.py`` script, the simulation loop is similar to the previous tutorial.
The only difference is that we create an instance of :class:`envs.RLTaskEnv` instead of the
:class:`envs.BaseEnv`. Consequently, now the :meth:`envs.RLTaskEnv.step` method returns additional signals
such as the reward and termination status. The information dictionary also maintains logging of quantities
such as the reward contribution from individual terms, the termination status of each term, the episode length etc.

.. literalinclude:: ../../../../source/standalone/tutorials/03_envs/run_cartpole_rl_env.py
   :language: python
   :pyobject: main
   :linenos:
   :lineno-start: 44


The Code Execution
~~~~~~~~~~~~~~~~~~

Similar to the previous tutorial, we can run the environment by executing the ``run_cartpole_rl_env.py`` script.

.. code-block:: bash

   ./orbit.sh -p source/standalone/tutorials/03_envs/run_cartpole_rl_env.py --num_envs 32


This should open a similar simulation as in the previous tutorial. However, this time, the environment
returns more signals that specify the reward and termination status. Additionally, the individual
environments reset themselves when they terminate based on the termination criteria specified in the
configuration.

To stop the simulation, you can either close the window, or press ``Ctrl+C`` in the terminal
where you started the simulation.

In this tutorial, we learnt how to create a task environment for reinforcement learning. We do this
by extending the base environment to include the rewards, terminations, commands and curriculum terms.
We also learnt how to use the :class:`envs.RLTaskEnv` class to run the environment and receive various
signals from it.

While it is possible to manually create an instance of :class:`envs.RLTaskEnv` class for a desired task,
this is not scalable as it requires specialized scripts for each task. Thus, we exploit the
:meth:`gymnasium.make` function to create the environment with the gym interface. We will learn how to do this
in the next tutorial.
