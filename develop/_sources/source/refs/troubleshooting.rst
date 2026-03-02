Tricks and Troubleshooting
==========================

.. note::

    The following lists some of the common tricks and troubleshooting methods that we use in our common workflows.
    Please also check the `troubleshooting page on Omniverse
    <https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/linux_troubleshooting.html>`__ for more
    assistance.


Debugging physics simulation stability issues
---------------------------------------------

When importing new robots into Isaac Lab or setting up a new environment, simulation instability
can often appear if the assets have not been tuned with reasonable simulation parameters.
In reinforcement learning scenarios, this will often result in NaNs propagating into the learning pipeline
due to invalid states in the simulation.

If this happens, we recommend consulting the
`Articulation and Robot Simulation Stability Guide <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/articulation_stability_guide.html>`_
which recommends various simulation parameters and best practices to achieve better stability in robot simulations.

Additionally, `Omniverse PhysX Visual Debugger <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/ux/source/omni.physx.pvd/docs/dev_guide/physx_visual_debugger.html>`_
allows for recording of data of PhysX simulations, which can often help simulation issues and aid the debugging process.

To enable OmniPVD capture in Isaac Lab, add the relevant kit arguments to the command line prompt when launching an Isaac Lab process

.. code:: bash

    ./isaaclab.sh -p scripts/demos/bipeds.py --kit_args "--/persistent/physics/omniPvdOvdRecordingDirectory=/tmp/ --/physics/omniPvdOutputEnabled=true" --headless


Checking the internal logs from the simulator
---------------------------------------------

When running the simulator from a standalone script, it logs warnings and errors to the terminal. At the same time,
it also logs internal messages to a file. These are useful for debugging and understanding the internal state of the
simulator. Depending on your system, the log file can be found in the locations listed
`here <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_faq.html#common-path-locations>`_.

To obtain the exact location of the log file, you need to check the first few lines of the terminal output when
you run the standalone script. The log file location is printed at the start of the terminal output. For example:

.. code:: bash

    [INFO] Using python from: /home/${USER}/git/IsaacLab/_isaac_sim/python.sh
    ...
    Passing the following args to the base kit application:  []
    Loading user config located at: '.../data/Kit/Isaac-Sim/2023.1/user.config.json'
    [Info] [carb] Logging to file: '.../logs/Kit/Isaac-Sim/2023.1/kit_20240328_183346.log'


In the above example, the log file is located at ``.../logs/Kit/Isaac-Sim/2023.1/kit_20240328_183346.log``,
``...`` is the path to the user's log directory. The log file is named ``kit_20240328_183346.log``

You can open this file to check the internal logs from the simulator. Also when reporting issues, please include
this log file to help us debug the issue.

Changing logging channel levels for the simulator
-------------------------------------------------

By default, the simulator logs messages at the ``WARN`` level and above on the terminal. You can change the logging
channel levels to get more detailed logs. The logging channel levels can be set through Omniverse's logging system.

To obtain more detailed logs, you can run your application with the following flags:

* ``--info``: This flag logs messages at the ``INFO`` level and above.
* ``--verbose``: This flag logs messages at the ``VERBOSE`` level and above.

For instance, to run a standalone script with verbose logging, you can use the following command:

.. code-block:: bash

    # Run the standalone script with info logging
    ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless --info

For more fine-grained control, you can modify the logging channels through the ``logger`` module.
For more information, please refer to its `documentation <https://docs.python.org/3/library/logging.html>`__.


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

    PhysX error: the application need to increase the PxgDynamicsMemoryConfig::foundLostPairsCapacity
    parameter to 3072, otherwise the simulation will miss interactions

In this case, you need to increase the size of the buffers passed to the
:class:`~isaaclab.sim.SimulationContext` class. The size of the buffers can be increased by setting
the :attr:`~isaaclab.sim.PhysxCfg.gpu_found_lost_pairs_capacity` parameter in the
:class:`~isaaclab.sim.PhysxCfg` class. For example, to increase the size of the buffers to
4096, you can use the following code:

.. code:: python

    import isaaclab.sim as sim_utils

    sim_cfg = sim_utils.SimulationConfig()
    sim_cfg.physx.gpu_found_lost_pairs_capacity = 4096
    sim = SimulationContext(sim_params=sim_cfg)

Please see the documentation for :class:`~isaaclab.sim.SimulationCfg` for more details
on the parameters that can be used to configure the simulation.


Preventing memory leaks in the simulator
----------------------------------------

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
    File "scripts/imitation_learning/robomimic/collect_demonstrations.py", line 166, in <module>
        main()
    File "scripts/imitation_learning/robomimic/collect_demonstrations.py", line 126, in main
        actions = pre_process_actions(delta_pose, gripper_command)
    File "scripts/imitation_learning/robomimic/collect_demonstrations.py", line 57, in pre_process_actions
        return torch.concat([delta_pose, gripper_vel], dim=1)
    TypeError: expected Tensor as element 1 in argument 0, but got int
    Exception ignored in: <function _make_registry.<locals>._Registry.__del__ at 0x7f94ac097f80>
    Traceback (most recent call last):
    File "../IsaacLab/_isaac_sim/kit/extscore/omni.kit.viewport.registry/omni/kit/viewport/registry/registry.py", line 103, in __del__
    File "../IsaacLab/_isaac_sim/kit/extscore/omni.kit.viewport.registry/omni/kit/viewport/registry/registry.py", line 98, in destroy
    TypeError: 'NoneType' object is not callable
    Exception ignored in: <function _make_registry.<locals>._Registry.__del__ at 0x7f94ac097f80>
    Traceback (most recent call last):
    File "../IsaacLab/_isaac_sim/kit/extscore/omni.kit.viewport.registry/omni/kit/viewport/registry/registry.py", line 103, in __del__
    File "../IsaacLab/_isaac_sim/kit/extscore/omni.kit.viewport.registry/omni/kit/viewport/registry/registry.py", line 98, in destroy
    TypeError: 'NoneType' object is not callable
    Exception ignored in: <function SettingChangeSubscription.__del__ at 0x7fa2ea173e60>
    Traceback (most recent call last):
    File "../IsaacLab/_isaac_sim/kit/kernel/py/omni/kit/app/_impl/__init__.py", line 114, in __del__
    AttributeError: 'NoneType' object has no attribute 'get_settings'
    Exception ignored in: <function RegisteredActions.__del__ at 0x7f935f5cae60>
    Traceback (most recent call last):
    File "../IsaacLab/_isaac_sim/extscache/omni.kit.viewport.menubar.lighting-104.0.7/omni/kit/viewport/menubar/lighting/actions.py", line 345, in __del__
    File "../IsaacLab/_isaac_sim/extscache/omni.kit.viewport.menubar.lighting-104.0.7/omni/kit/viewport/menubar/lighting/actions.py", line 350, in destroy
    TypeError: 'NoneType' object is not callable
    2022-12-02 15:41:54 [18,514ms] [Warning] [carb.audio.context] 1 contexts were leaked
    ../IsaacLab/_isaac_sim/python.sh: line 41: 414372 Segmentation fault      (core dumped) $python_exe "$@" $args
    There was an error running python

This is a known error with running standalone scripts with the Isaac Sim
simulator. Please scroll above the exceptions thrown with
``registry`` to see the actual error log.

In the above case, the actual error is:

.. code:: bash

    Traceback (most recent call last):
    File "scripts/imitation_learning/robomimic/tools/collect_demonstrations.py", line 166, in <module>
        main()
    File "scripts/imitation_learning/robomimic/tools/collect_demonstrations.py", line 126, in main
        actions = pre_process_actions(delta_pose, gripper_command)
    File "scripts/imitation_learning/robomimic/tools/collect_demonstrations.py", line 57, in pre_process_actions
        return torch.concat([delta_pose, gripper_vel], dim=1)
    TypeError: expected Tensor as element 1 in argument 0, but got int
