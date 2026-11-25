:orphan:

Recording Animations of Simulations
===================================

.. currentmodule:: isaaclab

Use Newton RTX to record simulation videos without Isaac Sim. The Kit-based USD animation
workflows are retained below for existing projects.

Newton RTX video recording
---------------------------

Install the rendering and video extras, then run the recording tutorial:

.. code-block:: bash

   uv run --extra ov --extra video python scripts/tutorials/07_visualizers/run_video_recording.py \
      --example 1 --num_envs 4 --viz newton_rtx

This records the Newton RTX viewport as an MP4 video. The tutorial also demonstrates recording
scene-camera outputs and multiple views. See :doc:`/source/concepts/visualization` for recorder
configuration. Video recording captures rendered frames; it does not produce the time-sampled
USD layers described below.

Isaac Sim / Kit USD animation (deprecated)
------------------------------------------

.. deprecated:: 3.1

   Isaac Lab's Kit recording paths will be removed in 3.2. Use Newton RTX for new video-recording
   workflows. The Stage Recorder and OVD instructions below remain available for projects that
   need their USD animation outputs.

Isaac Lab supports two approaches for recording animations of physics simulations: the **Stage Recorder** and the **OVD Recorder**.
Both generate USD outputs that can be played back in Omniverse, but they differ in how they work and when you’d use them.

The `Stage Recorder`_ extension listens to all motion and USD property changes in the stage during simulation
and records them as **time-sampled data**. The result is a USD file that captures only the animated changes—**not** the
full scene—and matches the hierarchy of the original stage at the time of recording.
This makes it easy to add as a sublayer for playback or rendering.

This method is built into Isaac Lab’s UI through the :class:`~isaaclab.envs.ui.BaseEnvWindow`.
However, to record the animation of a simulation, you need to disable `Fabric`_ to allow reading and writing
all the changes (such as motion and USD properties) to the USD stage.

The **OVD Recorder** is designed for more scalable or automated workflows. It uses OmniPVD to capture simulated physics from a played stage
and then **bakes** that directly into an animated USD file. It works with Fabric enabled and runs with CLI arguments.
The animated USD can be quickly replayed and reviewed by scrubbing through the timeline window, without simulating expensive physics operations.

.. note::

  Omniverse only supports **either** physics simulation **or** animation playback on a USD prim—never both at once.
  Disable physics on the prims you want to animate.


Stage Recorder
~~~~~~~~~~~~~~

In Isaac Lab, the Stage Recorder is integrated into the :class:`~isaaclab.envs.ui.BaseEnvWindow` class.
It’s the easiest way to capture physics simulations visually and works directly through the UI.

To record, Fabric must be disabled—this allows the recorder to track changes to USD and write them out.

Stage Recorder Settings
^^^^^^^^^^^^^^^^^^^^^^^

Isaac Lab sets up the Stage Recorder with sensible defaults in ``base_env_window.py``. If needed,
you can override or inspect these by using the Stage Recorder extension directly in Omniverse Create.

.. dropdown:: Settings used in base_env_window.py
  :icon: code

  .. literalinclude:: ../../../source/isaaclab/isaaclab/envs/ui/base_env_window.py
    :language: python
    :linenos:
    :pyobject: BaseEnvWindow._toggle_recording_animation_fn

Example Usage
^^^^^^^^^^^^^

Select a Kit-compatible environment and disable Fabric. This Cartpole example uses
``physics=isaacsim_physx`` and the ``env.sim.use_fabric=False`` configuration override:

.. code-block:: bash

   uv run --extra isaacsim python scripts/tutorials/03_envs/run_cartpole_rl_env.py \
      --num_envs 8 --device cpu --viz kit physics=isaacsim_physx env.sim.use_fabric=False

After launching, the Isaac Lab UI window will display a "Record Animation" button.
Click to begin recording. Click again to stop.

The following files are saved to the ``recordings/`` folder:

- ``Stage.usd`` — the original stage with physics disabled
- ``TimeSample_tk001.usd`` — the animation (time-sampled) layer

To play back, open Isaac Sim:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         uv run isaaclab -s

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         uv run isaaclab -s

Inside the Layers panel, insert both ``Stage.usd`` and ``TimeSample_tk001.usd`` as sublayers.
The animation will now play back when you hit the play button.

See the `tutorial on layering in Omniverse`_ for more on working with layers.


OVD Recorder
~~~~~~~~~~~~

The OVD Recorder uses OmniPVD to record simulation data and bake it directly into a new USD stage.
This method is more scalable and better suited for large-scale training scenarios (e.g. multi-env RL).

It’s not UI-controlled—the whole process is enabled through CLI flags and runs automatically.

.. note::

   The OVD Recorder uses OmniPVD, which only records **PhysX** simulations. If the active physics backend is not
   PhysX (for example, Newton, which is the default for many tasks), Isaac Lab raises an error at startup naming
   the active backend instead of recording. Select the PhysX backend by adding ``physics=isaacsim_physx`` to the
   command line, as shown below.

   The PhysX backend requires Isaac Sim. If it isn't installed yet, add ``--extra isaacsim`` to the ``uv run``
   command; see :ref:`installation-optional-extras` for details.


Workflow Summary
^^^^^^^^^^^^^^^^

1. User runs Isaac Lab with animation recording enabled via CLI
2. Isaac Lab starts simulation
3. OVD data is recorded as the simulation runs
4. At the specified stop time, the simulation is baked into an outputted USD file, and IsaacLab is closed
5. The final result is a fully baked, self-contained USD animation

Example Usage
^^^^^^^^^^^^^

To record an animation:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         uv run --extra isaacsim python scripts/tutorials/03_envs/run_cartpole_rl_env.py \
            --viz kit --anim_recording_enabled --anim_recording_start_time 1 \
            --anim_recording_stop_time 3 physics=isaacsim_physx

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         uv run --extra isaacsim python scripts\tutorials\03_envs\run_cartpole_rl_env.py ^
            --viz kit --anim_recording_enabled --anim_recording_start_time 1 ^
            --anim_recording_stop_time 3 physics=isaacsim_physx

.. note::

   The provided ``--anim_recording_stop_time`` should be greater than the simulation time.

.. warning::

   Currently, the final recording step can output many warning logs from [omni.usd]. This is a known issue, and these warning messages can be ignored.

After the stop time is reached, a file will be saved to:

.. code-block:: none

  anim_recordings/<timestamp>/baked_animation_recording.usda


.. _Stage Recorder: https://docs.omniverse.nvidia.com/extensions/latest/ext_animation_stage-recorder.html
.. _Fabric: https://docs.omniverse.nvidia.com/kit/docs/usdrt.scenegraph/latest/usd_fabric_usdrt.html
.. _Omniverse Launcher: https://docs.omniverse.nvidia.com/launcher/latest/index.html
.. _tutorial on layering in Omniverse: https://www.youtube.com/watch?v=LTwmNkSDh-c&ab_channel=NVIDIAOmniverse
