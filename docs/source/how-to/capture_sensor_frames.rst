.. _how-to-capture-sensor-frames:

Capturing sensor frames during training
=======================================

Isaac Lab supports saving image-like outputs from scene sensors during training using the
``CaptureEnvSensors`` Gymnasium wrapper. When ``--capture_env_sensors`` is set to a positive value,
Isaac Lab iterates over the environment's :class:`~isaaclab.scene.InteractiveScene` sensors and
writes each sensor's ``data.output`` tensors (for example, ``rgb``, ``depth``, or ``normals`` from a
:class:`~isaaclab.sensors.Camera`). The flag value is the number of parallel environment views to
tile into each saved frame grid. This differs from :doc:`record_video`, which records a single
perspective viewport clip of the scene.

The capture flags are registered on the shared training entrypoints for RSL-RL, RL-Games, Stable
Baselines3, and skrl. They are not available on ``play`` scripts.

This feature can be enabled using the following command line arguments with
``./isaaclab.sh train``:

* ``--capture_env_sensors``: number of parallel environments to include in each saved frame grid
  (default: ``0``, which disables capture)
* ``--capture_env_sensors_length``: length of each captured sensor frame window in **per-episode**
  steps (default: ``200``). Set to ``0`` to disable saving even when ``--capture_env_sensors`` is
  positive.
* ``--capture_env_sensors_interval``: interval between captured sensor frame windows in **per-episode**
  steps (default: ``2000``)
* ``--capture_env_sensors_format``: output format, either ``tensorboard`` (default) or ``file``

Enabling sensor capture automatically enables camera rendering during training, the same as
``--video``, which slows down both startup and runtime performance. Sensor capture can also be used
at the same time as ``--video`` when both per-sensor frames and a perspective viewport clip are
needed.

Example usage:

.. code-block:: shell

    ./isaaclab.sh train --rl_library rsl_rl --task=Isaac-Reorient-Cube-Shadow-Camera-Direct --capture_env_sensors 4 --capture_env_sensors_length 100 --capture_env_sensors_interval 2000 --capture_env_sensors_format file


The captured frames will be saved in the same directory as the training checkpoints, under
``logs/<rl_workflow>/<task>/<run>/sensor_frames/train``. Here ``<run>`` is the training run
directory (the same timestamped folder used for checkpoints and ``run.json``), not the per-episode
index used in output tags and file names.


Overview
--------

The sensor capture feature wraps the training environment through ``wrap_training_capture`` in
``isaaclab_rl.entrypoints.common`` and saves frames on reset and step when the current
**per-episode** step falls inside a capture window. For each image-like scene sensor, the wrapper:

* reads ``sensor.data.output`` and skips sensors whose output is not a dictionary of tensors
* resolves each output to a :class:`torch.Tensor`, including
  :class:`~isaaclab.utils.warp.ProxyArray` buffers through their ``.torch`` accessor
* selects the first ``capture_num_envs`` parallel environments from each tensor
* replaces non-finite values with zero before display normalization
* normalizes each output for display using :func:`~isaaclab.utils.images.normalize_camera_output_for_display`
  and tiles the selected views into a single image grid with
  :func:`~isaaclab.utils.images.make_camera_output_grid`
* writes the grid to TensorBoard or to disk as a PNG, depending on ``--capture_env_sensors_format``

Sensors with ``None`` for a given data type are skipped. Only tasks that define scene sensors with
image-like outputs (such as camera-based RL environments) produce captured frames.


Command-line options
--------------------

The training entrypoints register the capture flags in ``add_common_train_args``:

.. literalinclude:: ../../../source/isaaclab_rl/isaaclab_rl/entrypoints/common.py
   :language: python
   :lines: 277-300


Capture schedule
----------------

A frame is saved on every environment reset and after each step while the current **per-episode**
step is inside an active capture window. The episode step counter resets to ``0`` on every
environment reset. A step is inside the window when:

.. code-block:: text

    step % capture_env_sensors_interval < capture_env_sensors_length

For example, with ``--capture_env_sensors_length 100`` and ``--capture_env_sensors_interval 2000``,
the wrapper saves episode steps ``0-99`` after each reset. If an episode lasts longer than 2000
steps, another capture window opens at episode steps ``2000-2099``, then ``4000-4099``, and so on.

.. note::

   This schedule uses a **per-episode** step counter. :doc:`record_video` instead keys off the
   Gymnasium ``RecordVideo`` wrapper's **global** environment-step counter across episodes.


Episode indexing
^^^^^^^^^^^^^^^^

Each environment reset increments an episode index that starts at ``1`` on the first reset. Captured
frames are grouped by this index so consecutive episodes can be compared:

* TensorBoard tags use ``<sensor_name>/<data_type>/episode_<index>`` with a five-digit zero-padded
  episode index (for example, ``episode_00001``)
* File output uses ``episode_<index>_step_<step>.png`` under each sensor and data-type directory,
  with a five-digit episode index and an eight-digit step index (for example,
  ``episode_00001_step_00000042.png``)

The ``<step>`` value is the per-episode step counter (reset to ``0`` on every environment reset).
In TensorBoard, images are logged with ``global_step`` set to the total number of environment steps
since training started.


Image processing
----------------

Before tiling, each selected environment view is passed through
:func:`~isaaclab.utils.images.normalize_camera_output_for_display`, which maps common camera data
types to a ``[0, 1]`` float range suitable for PNG export:

* RGB-like outputs are scaled by ``255``
* Depth-like outputs (``depth``, ``distance_to_camera``, ``distance_to_image_plane``) are scaled by
  their per-frame maximum
* ``albedo`` keeps the first three channels and scales by ``255``
* ``normals`` are remapped from ``[-1, 1]`` to ``[0, 1]``

:func:`~isaaclab.utils.images.make_camera_output_grid` arranges the ``capture_num_envs`` views into a
roughly square grid (``nrow = round(sqrt(num_envs))``) before the image is written.


Output formats
--------------

**TensorBoard (default).** Each sensor output is logged as a separate image series per episode with
the tag ``<sensor_name>/<data_type>/episode_<index>``. View the captures alongside other training
metrics by pointing TensorBoard at the training log directory or the ``sensor_frames/train``
subdirectory:

.. code-block:: shell

    ./isaaclab.sh -p -m tensorboard.main --logdir logs/rsl_rl/Isaac-Reorient-Cube-Shadow-Camera-Direct/<run>/sensor_frames/train

**File.** PNG images are written per sensor, data type, episode, and step:

``<sensor_name>/<data_type>/episode_<index>_step_<step>.png``

Sensor and data-type names are sanitized for file output so they are safe as path components. For
example, a frame from the first episode at step ``42`` for sensor ``front/camera`` and data type
``rgb`` is written as ``front_camera/rgb/episode_00001_step_00000042.png``.


Summary
-------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - ``--capture_env_sensors_format``
     - Output location
     - How to view
   * - ``tensorboard`` (default)
     - ``sensor_frames/train`` event files
     - TensorBoard image tab (tagged by sensor, data type, and episode index)
   * - ``file``
     - ``sensor_frames/train/<sensor>/<data_type>/episode_<index>_step_<step>.png``
     - Any image viewer or filesystem browser


See also
--------

* :doc:`record_video` - record a perspective viewport clip with ``--video``
* :doc:`/source/overview/reinforcement-learning/rl_existing_scripts` - training entrypoints that
  expose the capture flags
* :doc:`/source/overview/core-concepts/visualization` - visualizers and rendering during training
* :doc:`/source/overview/core-concepts/sensors/camera` - camera sensors and annotator data types
* :doc:`proxy_array` - dual CPU/GPU sensor buffers read through ``.torch``
