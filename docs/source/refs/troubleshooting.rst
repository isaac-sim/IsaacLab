Tricks and Troubleshooting
==========================

Using CPU Scaling Governor for performance
------------------------------------------

By default on many systems, the CPU frequency governor is set to
“powersave” mode, which sets the CPU to lowest static frequency. To
increase the maximum performance, we recommend setting the CPU frequency
governor to “performance” mode. For more details, please check the the
link
`here <https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/7/html/power_management_guide/cpufreq_governors>`__.

.. warning::
    We advice not to set the governor to “performance” mode on a system with poor
    cooling (such as laptops), since it may cause the system to overheat.

-  To view existing ``scaling_governor`` value per CPU:

   .. code:: bash

      cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

-  To change the governor to “performance” mode for each CPU:

   .. code:: bash

      echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor


Observing long load times at the start of the simulation
--------------------------------------------------------

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


Receiving a “PhysX error” when running simulation on GPU
--------------------------------------------------------

When using the GPU pipeline, the buffers used for the physics simulation are allocated on the GPU only
once at the start of the simulation. This means that they do not grow dynamically as the number of
collisions or objects in the scene changes. If the number of collisions or objects in the scene
exceeds the size of the buffers, the simulation will fail with an error such as the following:

.. code:: bash

    PhysX error: the application need to increase the PxgDynamicsMemoryConfig::foundLostPairsCapacity parameter to 3072, otherwise the simulation will miss interactions

In this case, you need to increase the size of the buffers passed to the :class:`SimulationContext` class.
The size of the buffers can be increased by setting the ``found_lost_pairs_capacity`` in the ``sim_params``
argument to the :class:`SimulationContext` class. For example, to increase the size of the buffers to
``4096``, you can use the following code:

.. code:: python

    from omni.isaac.core.simulation_context import SimulationContext

    sim = SimulationContext(sim_params={"gpu_found_lost_pairs_capacity": 4096})

These settings are also directly exposed through the :class:`PhysxCfg` class in the ``omni.isaac.orbit_envs``
extension, which can be used to configure the simulation engine. Please see the documentation for
:class:`PhysxCfg` for more details.


Understanding the error logs from crashes
-----------------------------------------

Many times the simulator crashes due to a bug in the implementation.
This swamps the terminal with exceptions, some of which are coming from
the python interpreter calling ``__del__()`` destructor of the
simulation application. These typically look like the following:

.. code:: bash

    ...

    [INFO]: Completed setting up the environment...

    Traceback (most recent call last):
    File "source/standalone/workflows/robomimic/collect_demonstrations.py", line 166, in <module>
        main()
    File "source/standalone/workflows/robomimic/collect_demonstrations.py", line 126, in main
        actions = pre_process_actions(delta_pose, gripper_command)
    File "source/standalone/workflows/robomimic/collect_demonstrations.py", line 57, in pre_process_actions
        return torch.concat([delta_pose, gripper_vel], dim=1)
    TypeError: expected Tensor as element 1 in argument 0, but got int
    Exception ignored in: <function _make_registry.<locals>._Registry.__del__ at 0x7f94ac097f80>
    Traceback (most recent call last):
    File "../orbit/_isaac_sim/kit/extscore/omni.kit.viewport.registry/omni/kit/viewport/registry/registry.py", line 103, in __del__
    File "../orbit/_isaac_sim/kit/extscore/omni.kit.viewport.registry/omni/kit/viewport/registry/registry.py", line 98, in destroy
    TypeError: 'NoneType' object is not callable
    Exception ignored in: <function _make_registry.<locals>._Registry.__del__ at 0x7f94ac097f80>
    Traceback (most recent call last):
    File "../orbit/_isaac_sim/kit/extscore/omni.kit.viewport.registry/omni/kit/viewport/registry/registry.py", line 103, in __del__
    File "../orbit/_isaac_sim/kit/extscore/omni.kit.viewport.registry/omni/kit/viewport/registry/registry.py", line 98, in destroy
    TypeError: 'NoneType' object is not callable
    Exception ignored in: <function SettingChangeSubscription.__del__ at 0x7fa2ea173e60>
    Traceback (most recent call last):
    File "../orbit/_isaac_sim/kit/kernel/py/omni/kit/app/_impl/__init__.py", line 114, in __del__
    AttributeError: 'NoneType' object has no attribute 'get_settings'
    Exception ignored in: <function RegisteredActions.__del__ at 0x7f935f5cae60>
    Traceback (most recent call last):
    File "../orbit/_isaac_sim/extscache/omni.kit.viewport.menubar.lighting-104.0.7/omni/kit/viewport/menubar/lighting/actions.py", line 345, in __del__
    File "../orbit/_isaac_sim/extscache/omni.kit.viewport.menubar.lighting-104.0.7/omni/kit/viewport/menubar/lighting/actions.py", line 350, in destroy
    TypeError: 'NoneType' object is not callable
    2022-12-02 15:41:54 [18,514ms] [Warning] [carb.audio.context] 1 contexts were leaked
    ../orbit/_isaac_sim/python.sh: line 41: 414372 Segmentation fault      (core dumped) $python_exe "$@" $args
    There was an error running python

This is a known error with running standalone scripts with the Isaac Sim
simulator. Please scroll above the exceptions thrown with
``registry`` to see the actual error log.

In the above case, the actual error is:

.. code:: bash

    Traceback (most recent call last):
    File "source/standalone/workflows/robomimic/tools/collect_demonstrations.py", line 166, in <module>
        main()
    File "source/standalone/workflows/robomimic/tools/collect_demonstrations.py", line 126, in main
        actions = pre_process_actions(delta_pose, gripper_command)
    File "source/standalone/workflows/robomimic/tools/collect_demonstrations.py", line 57, in pre_process_actions
        return torch.concat([delta_pose, gripper_vel], dim=1)
    TypeError: expected Tensor as element 1 in argument 0, but got int
