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

App-level deterministic rendering via ``AppLauncher``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``--deterministic`` flag is provided by :meth:`isaaclab.app.AppLauncher.add_app_launcher_args`.
After the simulation app starts, :class:`~isaaclab.app.app_launcher.AppLauncher` applies RTX/RTPT carb
settings via :meth:`~isaaclab.app.app_launcher.AppLauncher.apply_rtx_determinism_settings`.

**Strict PyTorch determinism** (calling :meth:`~isaaclab.utils.seed.configure_seed` with
``torch_deterministic=True`` when you pass ``--deterministic``) is wired into the RL training scripts
for **RL-Games**, **skrl**, **RSL-RL**, and **Stable-Baselines3**: each calls
:meth:`~isaaclab.utils.seed.configure_seed` after constructing its framework runner or agent object
so library initialization is not disturbed, then training proceeds with the requested global RNG and
optional PyTorch deterministic algorithms. Whether you need ``--deterministic`` at the app level
depends on the workload: **physics-only** simulation does not require it; **RTX** rendering
(non-minimal mode) does require it for reproducible imagery; **Newton** rendering does not require it.

To enable deterministic RTX settings from the app launcher, pass ``--deterministic``.

.. code-block:: bash

  ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Cartpole-Camera --enable_cameras --headless --deterministic

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
