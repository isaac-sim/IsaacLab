Tricks and Troubleshooting
==========================

.. seealso::

   This page is the source of truth for the ``isaaclab-setup-troubleshooting`` agent skill
   (`skills/user/setup-troubleshooting/ <../../../skills/user/setup-troubleshooting/SKILL.md>`__).
   When you change this page, update the skill so agent guidance stays in sync. See
   :doc:`/source/overview/developer-guide/agent_skills`.

.. note::

    The following lists some of the common tricks and troubleshooting methods that we use in our common workflows.
    Please also check the `troubleshooting page on Omniverse
    <https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/linux_troubleshooting.html>`__ for more
    assistance.

.. contents::
   :local:
   :depth: 2


Installation and imports
------------------------

An Isaac Lab package cannot be imported
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``ModuleNotFoundError`` naming an Isaac Lab package almost always means the command is
not running inside the Isaac Lab environment, rather than that the package was skipped at
install time. Most packages cannot be deselected at all.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Missing module
     - What it means
   * - ``isaaclab``, ``isaaclab_assets``, ``isaaclab_tasks``, ``isaaclab_physx``,
       ``isaaclab_ov``, ``isaaclab_newton``, ``isaaclab_rl``, ``isaaclab_visualizers``,
       ``isaaclab_contrib``, ``isaaclab_experimental``, ``isaaclab_ppisp``,
       ``isaaclab_tasks_experimental``
     - Core packages. Every ``isaaclab install`` run installs them as editable packages and
       there is no way to opt out, so the error points at the environment rather than at the
       install command.
   * - ``isaaclab_mimic``, ``isaaclab_teleop``
     - Optional packages. ``isaaclab install`` (equivalently ``isaaclab install all``)
       includes them; ``isaaclab install core`` does not.
   * - ``rsl_rl``, ``rl_games``, ``skrl``, ``stable_baselines3``
     - Reinforcement learning frameworks, installed by the ``rl`` feature. They are part of
       the default install; ``isaaclab install 'rl[rsl-rl]'`` installs a single framework.
   * - ``isaacsim``
     - Isaac Sim itself, which is never installed implicitly. See
       :ref:`troubleshooting-isaacsim-missing`.

First check whether the package resolves in the uv-managed environment, from the
repository root:

.. code-block:: bash

   uv run python -c "import isaaclab_tasks; print('ok')"

If that succeeds but your own command fails, the command is running against a different
interpreter. Re-run it through ``uv run`` from the repository root:

.. code-block:: bash

   uv run python scripts/environments/random_agent.py --task Isaac-Cartpole --num_envs 4

If the import fails under ``uv run`` as well, recreate the documented source-install
environment for your workflow.

.. note::

   The ``ov``, ``contrib`` and ``tetrahedralization`` features are deliberately excluded
   from the default install and must be requested explicitly, for example
   ``isaaclab install 'ov[ovrtx]'``.

.. _troubleshooting-isaacsim-missing:

Isaac Sim is not installed
~~~~~~~~~~~~~~~~~~~~~~~~~~

``ModuleNotFoundError: No module named 'isaacsim'`` means a script that requires Isaac Sim
was launched without it. Either install Isaac Sim:

.. code-block:: bash

   ./isaaclab.sh -i isaacsim

or run a Newton-based task, which does not need Kit:

.. code-block:: bash

   uv run isaaclab train --task Isaac-Cartpole physics=newton_mjwarp --visualizer newton

See :doc:`/source/setup/quickstart` for the full list of ``physics=`` and ``renderer=``
selectors and the extras each one requires.

Dependency version conflicts during installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During pip or uv installs, the package manager may print dependency warnings of the form
``<package> requires <version>, but <other-package> requires <version>``, where an Isaac
Lab package, an Isaac Sim package, or a third-party package declares an incompatible
constraint. Common examples include ``coverage``, ``packaging``, ``numpy``, or ``Pillow``
constraints reported between ``isaaclab``, ``isaacsim-kernel``, ``isaacsim-core``,
``nvidia-srl-usd``, and ``moviepy``.

These messages are generally benign when the install command completes successfully. They
usually reflect package metadata that is stricter or older than the versions bundled and
tested with Isaac Sim. Prefer starting from a fresh virtual environment and using the
installation commands in the Isaac Lab docs. If the resolver aborts with
``No solution found``, or the installation leaves missing modules at runtime, recreate the
environment and install the documented Isaac Sim version before installing Isaac Lab.

GLIBC is too old
~~~~~~~~~~~~~~~~

Isaac Sim pip packages require GLIBC 2.35 or newer. Check your version with
``ldd --version``. Ubuntu 22.04 and later satisfy this. On older distributions, use the
`binary installation <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html>`_
method for Isaac Sim instead.


PhysX backends
--------------

These entries apply to the ``physics=isaacsim_physx`` and ``physics=ovphysx`` backends.

Simulation instability with newly imported robots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When importing new robots into Isaac Lab or setting up a new environment, simulation instability
can often appear if the assets have not been tuned with reasonable simulation parameters.
In reinforcement learning scenarios, this will often result in NaNs propagating into the learning pipeline
due to invalid states in the simulation.

If this happens, we recommend consulting the
`Articulation and Robot Simulation Stability Guide <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/articulation_stability_guide.html>`_
which recommends various simulation parameters and best practices to achieve better stability in robot simulations.

Recording a simulation with OmniPVD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `Omniverse PhysX Visual Debugger <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/ux/source/omni.physx.pvd/docs/dev_guide/physx_visual_debugger.html>`_
allows for recording of data of PhysX simulations, which can often help diagnose simulation
issues and aid the debugging process.

To enable OmniPVD capture in Isaac Lab, add the relevant kit arguments to the command line
prompt when launching an Isaac Lab process:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code:: bash

          uv run --extra isaacsim python scripts/demos/bipeds.py --kit_args "--/persistent/physics/omniPvdOvdRecordingDirectory=/tmp/ --/physics/omniPvdOutputEnabled=true"


   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code:: bash

          ./isaaclab.sh -p scripts/demos/bipeds.py --kit_args "--/persistent/physics/omniPvdOvdRecordingDirectory=/tmp/ --/physics/omniPvdOutputEnabled=true"

GPU buffer capacity errors
~~~~~~~~~~~~~~~~~~~~~~~~~~

When using the GPU pipeline, the buffers used for the physics simulation are allocated on
the GPU only once at the start of the simulation. They do not grow dynamically as the number
of collisions or objects in the scene changes. If the scene exceeds the size of a buffer, the
simulation fails with an error such as:

.. code:: bash

    PhysX error: the application need to increase the PxgDynamicsMemoryConfig::foundLostPairsCapacity
    parameter to 3072, otherwise the simulation will miss interactions

Raise the matching field on the physics configuration for the backend you are running. The
field is named after the PhysX parameter — for the error above it is
:attr:`~isaaclab_physx.physics.PhysxCfg.gpu_found_lost_pairs_capacity`:

.. code:: python

    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext
    from isaaclab_physx.physics import PhysxCfg

    sim_cfg = sim_utils.SimulationCfg(physics=PhysxCfg(gpu_found_lost_pairs_capacity=2**22))
    sim = SimulationContext(sim_cfg)

The same field exists on :class:`~isaaclab_ov.physics.OvPhysxCfg` for the ``ovphysx``
backend. Inside a task, set it on the task's preset so that both PhysX backends stay in
sync:

.. code:: python

    from isaaclab.physics import PhysxAutoCfg
    from isaaclab.utils.configclass import configclass
    from isaaclab_ov.physics import OvPhysxCfg
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_tasks.utils import PresetCfg

    @configclass
    class MyPhysicsCfg(PresetCfg):
        isaacsim_physx = PhysxCfg(gpu_found_lost_pairs_capacity=2**22)
        ovphysx = OvPhysxCfg(gpu_found_lost_pairs_capacity=2**22)
        physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)

The defaults are already large, so only raise a capacity when PhysX explicitly asks for it.
Please see :class:`~isaaclab.sim.SimulationCfg` for the other parameters that configure the
simulation.


Newton backends
---------------

These entries apply to the ``physics=newton_mjwarp`` and ``physics=newton_kamino`` backends.

.. _simulation-performance-troubleshooting:

Simulation or training runs slower than expected
------------------------------------------------

Start with a representative profile before changing physics or rendering settings. The
:doc:`Nsight Systems profiling guide <../how-to/profile_with_nsys>` can help distinguish time spent
in physics, rendering, environment code, and policy inference.

For PhysX workloads, check the following common causes:

* **Unneeded visualization:** Commands that do not select a visualizer launch without a viewer by
  default. If a configuration would otherwise launch one, pass ``--viz none`` to disable it.
* **Excessive collision work:** Avoid duplicated or overlapping collision geometry and use the
  simplest collider that provides the required fidelity.
* **GPU collider fallbacks:** A warning that a convex mesh failed to cook as GPU-compatible means
  its collision handling falls back to the CPU. Replace the collider with a primitive or bounding
  box approximation when possible. A static triangle mesh is another option only when the geometry
  is not part of a dynamic rigid body.

For broader tuning guidance, including CPU/GPU selection, solver settings, rendering, sensors, and
CPU configuration, consult the maintained upstream guides:

* `Isaac Sim Performance Optimization Handbook
  <https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/sim_performance_optimization_handbook.html>`_
* `Omni Physics Simulation Performance Guide
  <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/physics-performance.html>`_

For Newton-specific performance and solver parameters, see the
:doc:`Newton physics documentation <../overview/core-concepts/physical-backends/newton/index>`.


Joints actuate in PhysX but not in a Newton-based backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Newton resolves target modes for joints covered by an Isaac Lab actuator
configuration before constructing the solver. For an
:class:`~isaaclab.actuators.ImplicitActuatorCfg`, stiffness-only, damping-only,
both-gain, and zero-gain configurations select position, velocity, combined
position/velocity, and effort modes respectively. ``None`` retains the
corresponding imported USD gain. Explicit actuator configurations use effort
mode because Isaac Lab computes their effort directly.

Joints not covered by an Isaac Lab actuator configuration retain their imported
USD target modes. Thus, zero-gain USD drives no longer require
:attr:`~isaaclab.sim.schemas.JointDrivePropertiesCfg.ensure_drives_exist` solely
to make a configured joint actuate in Newton. The option remains available for
workflows that need to author placeholder drives independently of an Isaac Lab
actuator configuration. See :ref:`import-new-asset-ensure-drives-exist` for
details.


Renderers and visualizers
-------------------------

Crash in ``libusd_tf`` / USD symbol collision with OVRTX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you see a crash involving ``libusd_tf-*.so`` and conflicting USD versions
(e.g. ``pxrInternal_v0_25_5`` vs ``pxrInternal_v0_25_11``):

1. Ensure ``LD_PRELOAD`` is set to ovrtx's ``libcarb.so`` and install the OVRTX
   runtime with ``./isaaclab.sh -i 'ov[ovrtx]'`` (see :ref:`modularized installation <installation-selective-install>`)
2. Ensure ``isaacsim`` / ``omniverse-kit`` is **not** installed in the same
   environment — their bundled USD libraries conflict with ovrtx's

The Newton visualizer window does not appear
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Newton interactive viewer needs ``imgui-bundle``, which is a base dependency of Isaac
Lab. In a uv-managed environment it is always present, so a missing window normally points
at the display rather than the package. If you are running in an environment that was not
built from the Isaac Lab dependency set, install it explicitly:

.. code-block:: bash

   uv pip install imgui-bundle

The viser visualizer serves a web UI instead of opening a window
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``--visualizer viser`` does not open a native window. Check the terminal for the served
URL; the default port is ``8080``, configurable through
:attr:`~isaaclab_visualizers.ViserVisualizerCfg.port`. The ``viser`` package ships in the
``viser`` extra.


Distributed training
--------------------

NCCL errors during multi-GPU training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On some Linux multi-GPU systems, distributed training may fail with
``CUDA error: an illegal memory access was encountered`` reported by ``ProcessGroupNCCL``.
For documented NCCL workarounds, see :ref:`multi-gpu-nccl-troubleshooting`.


Debugging and diagnostics
-------------------------

Checking the internal logs from the simulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When running a Kit-based workflow from a standalone script, the simulator logs warnings and
errors to the terminal. At the same time, it also logs internal messages to a file. These
are useful for debugging and understanding the internal state of the simulator. Depending
on your system, the log file can be found in the locations listed
`here <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_faq.html#common-path-locations>`_.

To obtain the exact location of the log file, check the first few lines of the terminal
output when you run the standalone script. The log file location is printed at the start of
the terminal output, on a line of the form:

.. code:: bash

    [Info] [carb] Logging to file: '.../logs/Kit/Isaac-Sim/<version>/kit_<timestamp>.log'

You can open this file to check the internal logs from the simulator. When reporting issues,
please include this log file to help us debug the issue.

Changing logging channel levels for the simulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the simulator logs messages at the ``WARN`` level and above on the terminal. You can change the logging
channel levels to get more detailed logs. The logging channel levels can be set through Omniverse's logging system.

To obtain more detailed logs, you can run your application with the following flags:

* ``--info``: This flag logs messages at the ``INFO`` level and above.
* ``--verbose``: This flag logs messages at the ``VERBOSE`` level and above.

For instance, to run a standalone script with verbose logging, you can use the following command:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

          # Run the standalone script with info logging
          uv run python scripts/tutorials/00_sim/create_empty.py --info

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code-block:: bash

          # Run the standalone script with info logging
          ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --info

For more fine-grained control, you can modify the logging channels through the ``logger`` module.
For more information, please refer to its `documentation <https://docs.python.org/3/library/logging.html>`__.

Understanding the error logs from crashes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a Kit-based script crashes, the terminal is often swamped with exceptions, many of
which come from the Python interpreter calling ``__del__()`` destructors as the simulation
application tears down. They look like this:

.. code:: bash

    ...

    [INFO]: Completed setting up the environment...

    Traceback (most recent call last):
      File "scripts/imitation_learning/robomimic/collect_demonstrations.py", line 166, in <module>
        main()
      File "scripts/imitation_learning/robomimic/collect_demonstrations.py", line 126, in main
        actions = pre_process_actions(delta_pose, gripper_command)
      File "scripts/imitation_learning/robomimic/collect_demonstrations.py", line 57, in pre_process_actions
        return torch.concat([delta_pose, gripper_vel], dim=1)
    TypeError: expected Tensor as element 1 in argument 0, but got int
    Exception ignored in: <function _make_registry.<locals>._Registry.__del__ at 0x7f94ac097f80>
    Traceback (most recent call last):
      File ".../omni/kit/viewport/registry/registry.py", line 103, in __del__
      File ".../omni/kit/viewport/registry/registry.py", line 98, in destroy
    TypeError: 'NoneType' object is not callable
    Exception ignored in: <function SettingChangeSubscription.__del__ at 0x7fa2ea173e60>
    Traceback (most recent call last):
      File ".../omni/kit/app/_impl/__init__.py", line 114, in __del__
    AttributeError: 'NoneType' object has no attribute 'get_settings'
    [Warning] [carb.audio.context] 1 contexts were leaked
    Segmentation fault (core dumped)
    There was an error running python

The teardown exceptions are noise. Scroll **above** the ``registry`` and
``Exception ignored in`` blocks to find the actual error — in the example above:

.. code:: bash

    Traceback (most recent call last):
      File "scripts/imitation_learning/robomimic/collect_demonstrations.py", line 166, in <module>
        main()
      File "scripts/imitation_learning/robomimic/collect_demonstrations.py", line 126, in main
        actions = pre_process_actions(delta_pose, gripper_command)
      File "scripts/imitation_learning/robomimic/collect_demonstrations.py", line 57, in pre_process_actions
        return torch.concat([delta_pose, gripper_vel], dim=1)
    TypeError: expected Tensor as element 1 in argument 0, but got int

Observing long load times at the start of the simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first time you run the simulator, it will take a long time to load up. This is because the
simulator is compiling shaders and loading assets. Subsequent runs should be faster to start up,
but may still take some time.

Please note that once the Isaac Sim app loads, the environment creation time may scale linearly with
the number of environments. Please expect a longer load time if running with thousands of
environments or if each environment contains a larger number of assets. We are continually working
on improving the time needed for this.

When an instance of Isaac Sim is already running, launching another Isaac Sim instance in a different
process may appear to hang at startup for the first time. Please be patient and give it some time as
the second process will take longer to start up due to slower shader compilation.

Preventing memory leaks in the simulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Memory leaks in the Isaac Sim simulator can occur when C++ callbacks are registered with Python objects.
This happens when callback functions within classes maintain references to the Python objects they are
associated with. As a result, Python's garbage collection is unable to reclaim memory associated with
these objects, preventing the corresponding C++ objects from being destroyed. Over time, this can lead
to memory leaks and increased resource usage.

To prevent memory leaks in the Isaac Sim simulator, it is essential to use weak references when registering
callbacks with the simulator. This ensures that Python objects can be garbage collected when they are no
longer needed, thereby avoiding memory leaks. The `weakref <https://docs.python.org/3/library/weakref.html>`_
module from the Python standard library can be employed for this purpose.

For example, consider a class with a callback function ``on_event_callback`` that needs to be registered
with the simulator. If you use a strong reference to the ``MyClass`` object when passing the callback,
the reference count of the ``MyClass`` object will be incremented. This prevents the ``MyClass`` object
from being garbage collected when it is no longer needed, i.e., the ``__del__`` destructor will not be
called.

.. code:: python

    import omni.kit

    class MyClass:
        def __init__(self):
            app_interface = omni.kit.app.get_app_interface()
            self._handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                self.on_event_callback
            )

        def __del__(self):
            self._handle.unsubscribe()
            self._handle = None

        def on_event_callback(self, event):
            # do something with the message


To fix this issue, it's crucial to employ weak references when registering the callback. While this approach
adds some verbosity to the code, it ensures that the ``MyClass`` object can be garbage collected when no longer
in use. Here's the modified code:

.. code:: python

    import omni.kit
    import weakref

    class MyClass:
        def __init__(self):
            app_interface = omni.kit.app.get_app_interface()
            self._handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                lambda event, obj=weakref.proxy(self): obj.on_event_callback(event)
            )

        def __del__(self):
            self._handle.unsubscribe()
            self._handle = None

        def on_event_callback(self, event):
            # do something with the message


In this revised code, the weak reference ``weakref.proxy(self)`` is used when registering the callback,
allowing the ``MyClass`` object to be properly garbage collected.

By following this pattern, you can prevent memory leaks and maintain a more efficient and stable simulation.
