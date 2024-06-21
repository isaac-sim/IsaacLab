.. _feature-workflows:


Task Design Workflows
=====================

.. currentmodule:: omni.isaac.lab

Environments define the interface between the agent and the simulation. In the simplest case, the environment provides
the agent with the current observations and executes the actions provided by the agent. In a Markov Decision Process
(MDP) formulation, the environment can also provide additional information such as the current reward, done flag, and
information about the current episode.

While the environment interface is simple to understand, its implementation can vary significantly depending on the
complexity of the task. In the context of reinforcement learning (RL), the environment implementation can be broken down
into several components, such as the reward function, observation function, termination function, and reset function.
Each of these components can be implemented in different ways depending on the complexity of the task and the desired
level of modularity.

We provide two different workflows for designing environments with the framework:

* **Manager-based**: The environment is decomposed into individual components (or managers) that handle different
  aspects of the environment (such as computing observations, applying actions, and applying randomization). The
  user defines configuration classes for each component and the environment is responsible for coordinating the
  managers and calling their functions.
* **Direct**: The user defines a single class that implements the entire environment directly without the need for
  separate managers. This class is responsible for computing observations, applying actions, and computing rewards.

Both workflows have their own advantages and disadvantages. The manager-based workflow is more modular and allows
different components of the environment to be swapped out easily. This is useful when prototyping the environment
and experimenting with different configurations. On the other hand, the direct workflow is more efficient and allows
for more fine-grained control over the environment logic. This is useful when optimizing the environment for performance
or when implementing complex logic that is difficult to decompose into separate components.


Manager-Based Environments
--------------------------

Manager-based environments promote modular implementations of reinforcement learning tasks
through the use of Managers. Each component of the task, such as rewards, observations, termination
can all be specified as individual configuration classes that are then passed to the corresponding
manager classes. Each manager is responsible for parsing the configurations and processing
the contents specified in each config class. The manager implementations are taken care of by
the base class :class:`envs.ManagerBasedRLEnv`.

With this approach, it is simple to switch implementations of some components in the task
while leaving the remaining of the code intact. This is desirable when collaborating with others
on implementing a reinforcement learning environment, where contributors may choose to use
different combinations of configurations for the reinforcement learning components of the task.

A class definition of a manager-based environment consists of defining a task configuration class that
inherits from :class:`envs.ManagerBasedRLEnvCfg`. This class should contain variables assigned to various
configuration classes for each of the components of the RL task, such as the ``ObservationCfg``
or ``RewardCfg``. The entry point of the environment becomes the base class :class:`envs.ManagerBasedRLEnv`,
which will process the main task config and iterate through the individual configuration classes that are defined
in the task config class.

.. dropdown:: Example for defining the reward function for the Cartpole task using the manager-style
    :icon: plus

    The following class is a part of the Cartpole environment configuration class. The :class:`RewardsCfg` class
    defines individual terms that compose the reward function. Each reward term is defined by its function
    implementation, weight and additional parameters to be passed to the function. Users can define multiple
    reward terms and their weights to be used in the reward function.

    .. literalinclude:: ../../../source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/cartpole/cartpole_env_cfg.py
        :language: python
        :pyobject: RewardsCfg

.. seealso::

    We provide a more detailed tutorial for setting up an environment using the manager-based workflow at
    :ref:`tutorial-create-manager-rl-env`.


Direct Environments
-------------------

The direct-style environment more closely aligns with traditional implementations of reinforcement learning environments,
where a single script implements the reward function, observation function, resets, and all other components
of the environment. This approach does not use the Manager classes. Instead, users are left with the freedom
to implement the APIs from the base class :class:`envs.DirectRLEnv`. For users migrating from the IsaacGymEnvs
or OmniIsaacGymEnvs framework, this workflow will have a closer implementation to the previous frameworks.

When defining an environment following the direct-style implementation, a task configuration class inheriting from
:class:`envs.DirectRLEnvCfg` is used for defining task environment configuration variables, such as the number
of observations and actions. Adding configuration classes for the managers are not required and will not be processed
by the base class. In addition to the configuration class, the logic of the task should be defined in a new
task class that inherits from the base class :class:`envs.DirectRLEnv`. This class will then implement the main
task logics, including setting up the scene, processing the actions, computing resets, rewards, and observations.

This approach may bring more performance benefits for the environment, as it allows implementing large chunks
of logic with optimized frameworks such as `PyTorch JIT <https://pytorch.org/docs/stable/jit.html>`_ or
`Warp <https://github.com/NVIDIA/warp>`_. This may be important when scaling up training for large and complex
environments. Additionally, data may be cached in class variables and reused in multiple APIs for the class.
This method provides more transparency in the implementations of the environments, as logic is defined
within the task class instead of abstracted with the use the Managers.

.. dropdown:: Example for defining the reward function for the Cartpole task using the direct-style
    :icon: plus

    The following function is a part of the Cartpole environment class and is responsible for computing the rewards.

    .. literalinclude:: ../../../source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/cartpole/cartpole_env.py
        :language: python
        :pyobject: CartpoleEnv._get_rewards
        :dedent: 4

    It calls the :meth:`compute_rewards` function which is Torch JIT compiled for performance benefits.

    .. literalinclude:: ../../../source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/cartpole/cartpole_env.py
        :language: python
        :pyobject: compute_rewards

.. seealso::

    We provide a more detailed tutorial for setting up a RL environment using the direct workflow at
    :ref:`tutorial-create-direct-rl-env`.
