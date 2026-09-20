:orphan:

.. _how-to-estimate-how-cameras-can-run:


Find How Many/What Cameras You Should Train With
================================================

.. currentmodule:: isaaclab

Isaac Lab provides two camera implementations with different capabilities: the renderer-backed
:class:`~isaaclab.sensors.Camera` and the geometry-based :class:`~isaaclab.sensors.RayCasterCamera`.
``TiledCamera`` is now a deprecated alias of ``Camera`` rather than a separate implementation. The
``benchmark_cameras.py`` script characterizes the sensor implementations at different camera counts,
image dimensions, and data types.

This utility is provided so that one easily can find the camera type/parameters that are the most performant
while meeting the requirements of the user's scenario. This utility also helps estimate
the maximum number of cameras one can realistically run, assuming that one wants to maximize the number
of environments while minimizing step time.

This utility can inject cameras into an existing task from the gym registry,
which can be useful for benchmarking cameras in a specific scenario. Also,
if you install ``nvidia-ml-py``, you can let this utility automatically find the maximum
number of cameras that can run in your task environment up to a
certain specified system resource utilization threshold (without training; taking zero actions
at each timestep).

This guide accompanies the ``benchmark_cameras.py`` script in the ``scripts/benchmarks``
directory.

.. dropdown:: Code for benchmark_cameras.py
   :icon: code

   .. literalinclude:: ../../../scripts/benchmarks/benchmark_cameras.py
      :language: python
      :linenos:


Possible Parameters
-------------------

First, run

.. code-block:: bash

   uv run python scripts/benchmarks/benchmark_cameras.py --help

to see all possible parameters you can vary with this utility.


See the command line parameters related to ``autotune`` for more information about
automatically determining maximum camera count.


Compare Performance in Task Environments and Automatically Determine Task Max Camera Count
------------------------------------------------------------------------------------------

For example, to see how your system handles 100 renderer-backed cameras in
the cartpole environment, with 2 cameras per environment (so 50 environments total)
only in RGB mode, run

.. code-block:: bash

   uv run python scripts/benchmarks/benchmark_cameras.py \
      --task Isaac-Cartpole --num_cameras 100 --task_num_cameras_per_env 2 --camera_data_types rgb

If you have ``nvidia-ml-py`` installed (``uv pip install nvidia-ml-py``), you can also
find the maximum number of cameras that you could run in the specified environment up to
a certain performance threshold (specified by max CPU utilization percent, max RAM utilization percent,
max GPU compute percent, and max GPU memory percent). For example, to find the maximum number of cameras
you can run with cartpole, you could run:

.. code-block:: bash

   uv run python scripts/benchmarks/benchmark_cameras.py \
      --task Isaac-Cartpole --num_cameras 100 --task_num_cameras_per_env 2 \
      --camera_data_types rgb --autotune --autotune_max_percentage_util 100 80 50 50

Autotune may lead to the program crashing, which means that it tried to run too many cameras at once.
However, the max percentage utilization parameter is meant to prevent this from happening.

The output of the benchmark doesn't include the overhead of training the network, so consider
decreasing the maximum utilization percentages to account for this overhead. The final output camera
count is for all cameras, so to get the total number of environments, divide the output camera count
by the number of cameras per environment.


Compare Camera Implementations (Without a Specified Task)
----------------------------------------------------------

This tool can also assess performance without a task environment.
For example, to view 100 random objects with 2 renderer-backed cameras, run

.. code-block:: bash

   uv run python scripts/benchmarks/benchmark_cameras.py \
      --height 100 --width 100 --num_cameras 2 \
      --camera_data_types instance_segmentation normals --num_objects 100 --experiment_length 100

To benchmark ray casting against the ground plane instead, run

.. code-block:: bash

   uv run python scripts/benchmarks/benchmark_cameras.py \
      --num_ray_caster_cameras 2 --camera_data_types distance_to_image_plane

If your system cannot handle this due to performance reasons, then the process will be killed.
It's recommended to monitor CPU/RAM utilization and GPU utilization while running this script, to get
an idea of how many resources rendering the desired camera requires. In Ubuntu, you can use tools like ``htop`` and ``nvtop``
to live monitor resources while running this script, and in Windows, you can use the Task Manager.

If your system has a hard time handling the desired cameras, you can try the following

   - Switch to headless mode (omit ``--viz``, or pass ``--viz none`` if a config selects visualizers).
   - Ensure you are using the GPU pipeline rather than CPU.
   - Use :class:`~isaaclab.sensors.RayCasterCamera` when rendered appearance is not required.
   - Decrease camera resolution.
   - Decrease the number of camera data types.
   - Decrease the number of cameras.
   - Decrease the number of objects in the scene.

If your system is able to handle the amount of cameras, then the time statistics will be printed to the terminal.
The script exits after the requested warmup and measured steps.
