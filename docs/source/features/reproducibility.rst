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
is set to a fixed value using the :meth:`~isaacsim.core.utils.torch.set_seed` method. This method sets the
random seed for both the CPU and GPU globally across different libraries, including PyTorch and
NumPy.

In the included workflow scripts, the seed specified in the learning agent's configuration file or the
command line argument is used to set the random seed for the environment. This ensures that the
simulation results are reproducible across different runs. The seed is set into the environment
parameters :attr:`isaaclab.envs.ManagerBasedEnvCfg.seed` or :attr:`isaaclab.envs.DirectRLEnvCfg.seed`
depending on the manager-based or direct environment implementation respectively.

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
