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

The ``--deterministic`` flag
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``--deterministic`` flag is provided by :meth:`isaaclab.app.AppLauncher.add_app_launcher_args`.
:class:`~isaaclab.app.app_launcher.AppLauncher` publishes ``/isaaclab/render/deterministic``.
The Isaac RTX backend reads it on init and applies
:func:`isaaclab_physx.renderers.isaac_rtx_renderer_utils.apply_isaac_rtx_determinism_settings`.

**Strict PyTorch determinism** (calling :meth:`~isaaclab.utils.seed.configure_seed` with
``torch_deterministic=True`` when you pass ``--deterministic``) is wired into the RL training entrypoints
for **RL-Games**, **skrl**, **RSL-RL**, and **Stable-Baselines3**: each calls
:meth:`~isaaclab.utils.seed.configure_seed` after constructing its framework runner or agent object
so library initialization is not disturbed, then training proceeds with the requested global RNG and
optional PyTorch deterministic algorithms. Whether the **rendering** half of the flag matters depends
on the workload: **physics-only** simulation does not render at all; **RTX** rendering (non-minimal
mode) needs it for reproducible imagery; **Newton** rendering is already deterministic.

.. note::

   For convolutional workloads, setting ``TORCH_CUDNN_V8_API_DISABLED=1`` before launching training
   may improve run-to-run determinism by making PyTorch use the cuDNN v7 API instead of cuDNN v8
   execution plans.

**Physics determinism** comes from the same flag in the Isaac Lab RL training entrypoints, which
set :attr:`~isaaclab.physics.PhysicsCfg.deterministic` on the backend resolved by presets. That
field is the backend-agnostic request: each physics manager translates it into its own settings
when the simulation starts, and raises when its configuration cannot provide the guarantee. Newton
selects ``deterministic_mode="run_to_run"`` and applies the MJWarp prerequisite; PhysX and OvPhysX
enable enhanced determinism, which is best-effort on OvPhysX and not verified end to end. Set
:attr:`~isaaclab.physics.PhysicsCfg.deterministic` directly to get the same behavior from a script
that does not use the RL entrypoints; ``--deterministic`` alone configures Torch and rendering
only.

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

        uv run --extra rl-games isaaclab train --rl_library rl_games \
          --task Isaac-Cartpole-Camera --deterministic

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code-block:: bash

        ./isaaclab.sh train --rl_library rl_games \
          --task Isaac-Cartpole-Camera --deterministic

Newton physics determinism
^^^^^^^^^^^^^^^^^^^^^^^^^^

Set :attr:`isaaclab_newton.physics.NewtonCfg.deterministic_mode` to
``"gpu_to_gpu"`` to request reproducibility across GPU architectures, or to
``"run_to_run"`` to request reproducibility on one GPU. Newton applies the
selected mode to supported solver kernels and enables deterministic contact
ordering in its collision pipeline. Deterministic execution can increase
memory use and reduce simulation performance. MJWarp on the GPU with
:attr:`isaaclab_newton.physics.MJWarpSolverCfg.disable_sensors` set to ``True``,
XPBD, and Featherstone are supported; selecting an unsupported solver raises
an error. MuJoCo on the CPU
(:attr:`isaaclab_newton.physics.MJWarpSolverCfg.use_mujoco_cpu`) is already
reproducible and is left unchanged. Set this attribute directly to request the
stronger ``"gpu_to_gpu"`` guarantee, which takes precedence over
:attr:`isaaclab.physics.PhysicsCfg.deterministic`. Setting it to
``"not_guaranteed"`` does not opt out, because that is indistinguishable from
the default; clear :attr:`isaaclab.physics.PhysicsCfg.deterministic` instead.

.. note::

   ``disable_sensors`` is required rather than optional: MuJoCo Warp's tactile
   sensor kernel applies two atomic reduction families to one output array, which
   Warp's deterministic code generation cannot lower, so the sensor module fails
   to compile under a determinism guarantee. Disabling it also skips the
   ``rne_postconstraint`` stage, which fills the Newton ``body_qdd`` and
   ``body_parent_f`` state. The IMU, PVA, and joint-wrench sensors read that
   state, so Newton raises rather than let them report stale values: remove those
   sensors, or drop the determinism request. Integrations
   that consume native MJWarp sensor outputs directly are affected too, and are
   not detected.

.. warning::

   Deterministic contact ordering adds sorting work and allocates buffers sized
   for the configured maximum contact count. Runtime and memory overhead
   therefore grow with contact capacity. Enable this mode only when its
   reproducibility guarantee is required.

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
