.. _how-to-estimate-how-cameras-can-run:


Find How Many/What Cameras You Should Train With
================================================

.. currentmodule:: isaaclab

Currently in Isaac Lab, there are several camera types; USD Cameras (standard), Tiled Cameras,
and Ray Caster cameras. These camera types differ in functionality and performance. The ``benchmark_cameras.py``
script can be used to understand the difference in cameras types, as well to characterize their relative performance
at different parameters such as camera quantity, image dimensions, and data types.

This utility is provided so that one easily can find the camera type/parameters that are the most performant
while meeting the requirements of the user's scenario. This utility also helps estimate
the maximum number of cameras one can realistically run, assuming that one wants to maximize the number
of environments while minimizing step time.

This utility can inject cameras into an existing task from the gym registry,
which can be useful for benchmarking cameras in a specific scenario. Also,
if you install ``pynvml``, you can let this utility automatically find the maximum
numbers of cameras that can run in your task environment up to a
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

   ./isaaclab.sh -p scripts/benchmarks/benchmark_cameras.py -h

to see all possible parameters you can vary with this utility.


See the command line parameters related to ``autotune`` for more information about
automatically determining maximum camera count.


Compare Performance in Task Environments and Automatically Determine Task Max Camera Count
------------------------------------------------------------------------------------------

Currently, tiled cameras are the most performant camera that can handle multiple dynamic objects.

For example, to see how your system could handle 100 tiled cameras in
the cartpole environment, with 2 cameras per environment (so 50 environments total)
only in RGB mode, run

.. code-block:: bash

   ./isaaclab.sh -p scripts/benchmarks/benchmark_cameras.py \
   --task Isaac-Cartpole-v0 --num_tiled_cameras 100 \
   --task_num_cameras_per_env 2 \
   --tiled_camera_data_types rgb

If you have pynvml installed, (``./isaaclab.sh -p -m pip install pynvml``), you can also
find the maximum number of cameras that you could run in the specified environment up to
a certain performance threshold (specified by max CPU utilization percent, max RAM utilization percent,
max GPU compute percent, and max GPU memory percent). For example, to find the maximum number of cameras
you can run with cartpole, you could run:

.. code-block:: bash

   ./isaaclab.sh -p scripts/benchmarks/benchmark_cameras.py \
   --task Isaac-Cartpole-v0 --num_tiled_cameras 100 \
   --task_num_cameras_per_env 2 \
   --tiled_camera_data_types rgb --autotune \
   --autotune_max_percentage_util 100 80 50 50

Autotune may lead to the program crashing, which means that it tried to run too many cameras at once.
However, the max percentage utilization parameter is meant to prevent this from happening.

The output of the benchmark doesn't include the overhead of training the network, so consider
decreasing the maximum utilization percentages to account for this overhead. The final output camera
count is for all cameras, so to get the total number of environments, divide the output camera count
by the number of cameras per environment.


Compare Camera Type and Performance (Without a Specified Task)
--------------------------------------------------------------

This tool can also asses performance without a task environment.
For example, to view 100 random objects with 2 standard cameras, one could run

.. code-block:: bash

   ./isaaclab.sh -p scripts/benchmarks/benchmark_cameras.py \
   --height 100 --width 100 --num_standard_cameras 2 \
   --standard_camera_data_types instance_segmentation_fast normals --num_objects 100 \
   --experiment_length 100

If your system cannot handle this due to performance reasons, then the process will be killed.
It's recommended to monitor CPU/RAM utilization and GPU utilization while running this script, to get
an idea of how many resources rendering the desired camera requires. In Ubuntu, you can use tools like ``htop`` and ``nvtop``
to live monitor resources while running this script, and in Windows, you can use the Task Manager.

If your system has a hard time handling the desired cameras, you can try the following

   - Switch to headless mode (supply ``--headless``)
   - Ensure you are using the GPU pipeline not CPU!
   - If you aren't using Tiled Cameras, switch to Tiled Cameras
   - Decrease camera resolution
   - Decrease how many data_types there are for each camera.
   - Decrease the number of cameras
   - Decrease the number of objects in the scene

If your system is able to handle the amount of cameras, then the time statistics will be printed to the terminal.
After the simulations stops it can be closed with CTRL+C.
