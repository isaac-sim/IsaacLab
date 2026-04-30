Reproducibility and Determinism
-------------------------------

Given the same hardware and Isaac Sim (and consequently PhysX) version, the simulation produces
identical results for scenes with rigid bodies and articulations. However, the simulation results can
vary across different hardware configurations due to floating point precision and rounding errors.
At present, PhysX does not guarantee determinism for any scene with non-rigid bodies, such as cloth
or soft bodies. For more information, please refer to the `PhysX Determinism documentation`_.

Based on above, Isaac Lab provides a deterministic simulation that ensures consistent simulation
results across different runs. This is achieved by using the same random seed for the
simulation environment and the physics engine. At construction of the environment, the random seed
is set to a fixed value using the :meth:`~isaaclab.utils.seed.configure_seed` method. This method sets the
random seed for both the CPU and GPU globally across different libraries, including PyTorch and
NumPy.

In the included workflow scripts, the seed specified in the learning agent's configuration file or the
command line argument is used to set the random seed for the environment. This ensures that the
simulation results are reproducible across different runs. The seed is set into the environment
parameters :attr:`isaaclab.envs.ManagerBasedEnvCfg.seed` or :attr:`isaaclab.envs.DirectRLEnvCfg.seed`
depending on the manager-based or direct environment implementation respectively.

.. note::

   **Strict PyTorch determinism is only wired into the RL-Games training script.**

   The optional ``--deterministic`` flag (which calls :meth:`~isaaclab.utils.seed.configure_seed` with
   ``torch_deterministic=True``, enabling ``torch.use_deterministic_algorithms(True)`` and related CUDNN
   settings) exists only on ``scripts/reinforcement_learning/rl_games/train.py``.

   The RSL-RL, Stable-Baselines3, SKRL, and RLinf training scripts under
   ``scripts/reinforcement_learning/`` still honor ``--seed`` / agent configuration for the Isaac Lab
   environment and learning stack, but they do **not** expose an equivalent opt-in for strict PyTorch-wide
   deterministic algorithms. If you need that behavior with another framework, call
   :meth:`~isaaclab.utils.seed.configure_seed` with ``torch_deterministic=True`` from your own training
   entry point, keeping in mind that some CUDA operations may error when no deterministic implementation
   exists.

To enable deterministic rendering/app settings, launch workflows with the deterministic experience file:

.. code-block:: bash

  ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Cartpole-v0 \
    --experience isaaclab.python.headless.determinism.kit

For RL-Games, combine this with ``--deterministic`` if you also want strict PyTorch deterministic
algorithms in addition to deterministic app/render settings.

For results on our determinacy testing for RL training, please check the GitHub Pull Request `#940`_.

.. tip::

  Due to GPU work scheduling, there's a possibility that runtime changes to simulation parameters
  may alter the order in which operations take place. This occurs because environment updates can
  happen while the GPU is occupied with other tasks. Due to the inherent nature of floating-point
  numeric storage, any modification to the execution ordering can result in minor changes in the
  least significant bits of output data. These changes may lead to divergent execution over the
  course of simulating thousands of environments and simulation frames.

  An illustrative example of this issue is observed with the runtime domain randomization of object's
  physics materials. This process can introduce both determinacy and simulation issues when executed
  on the GPU due to the way these parameters are passed from the CPU to the GPU in the lower-level APIs.
  Consequently, it is strongly advised to perform this operation only at setup time, before the
  environment stepping commences.


.. _PhysX Determinism documentation: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/API.html#determinism
.. _#940: https://github.com/isaac-sim/IsaacLab/pull/940
