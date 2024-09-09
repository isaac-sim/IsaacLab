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
is set to a fixed value using :meth:`omni.isaac.core.utils.torch.set_seed`. This method sets the
random seed for both the CPU and GPU globally across different libraries, including PyTorch and
NumPy.

.. _PhysX Determinism documentation: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/API.html#determinism
