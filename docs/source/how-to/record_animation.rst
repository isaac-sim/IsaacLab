:orphan:

Recording Animations of Simulations
===================================

.. currentmodule:: isaaclab

Use Newton RTX to record simulation videos or kit-less OVPhysX to capture live physics data.
Existing projects can also use the Kit recorders to export USD animations.

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

The **Stage Recorder** and the **Isaac Sim OVD Recorder** generate USD animations for playback in Omniverse.

The `Stage Recorder`_ extension listens to all motion and USD property changes in the stage during simulation
and records them as **time-sampled data**. The result is a USD file that captures only the animated changes—**not** the
full scene—and matches the hierarchy of the original stage at the time of recording.
This makes it easy to add as a sublayer for playback or rendering.

This method is built into Isaac Lab’s UI through the :class:`~isaaclab.envs.ui.BaseEnvWindow`.
However, to record the animation of a simulation, you need to disable `Fabric`_ to allow reading and writing
all the changes (such as motion and USD properties) to the USD stage.

The **Isaac Sim OVD Recorder** uses OmniPVD to capture simulated physics and bake it into an animated USD file.
It works with Fabric enabled and runs with CLI arguments.
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

The animation recording CLI uses the Isaac Sim PhysX backend to capture a single interval and bake it
into a new USD stage. Use ``--viz kit`` to open a window and leave ``HEADLESS`` unset or set to ``0``.

.. note::

   Select Isaac Sim PhysX by adding ``physics=isaacsim_physx`` to the command line, as shown below.
   Kit-less OVPhysX uses the live capture API described below.

   The examples include the Isaac Sim extra. See :ref:`installation-optional-extras` for details.


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

   Set ``--anim_recording_stop_time`` greater than ``--anim_recording_start_time`` and let the process run
   through the export.

.. warning::

   Currently, the final recording step can output many warning logs from [omni.usd]. This is a known issue, and these warning messages can be ignored.

After the stop time is reached, a file will be saved to:

.. code-block:: none

  anim_recordings/<timestamp>/baked_animation_recording.usda


Live OVD capture with kit-less OVPhysX
--------------------------------------

Use ``OvPhysxCfg(ovd_recording_capable=True)`` before the first native PhysX instance is created. The
default is ``False``. After the scene's initial reset completes, get the native OVPhysX handle with
``sim.backend_manager.get_physx_instance()``. Call ``start_recording()`` with an OVD destination and
``stop_recording()`` on the stepping thread between completed steps.

For multiple captures in a job, let the simulation code choose when to start and stop each capture.
Only one capture per process can be active at a time. After stopping, the simulation can keep stepping
and start another capture with a different destination. The live API writes OVD data, and the
``--anim_recording_*`` flags are for the Isaac Sim USD-baking workflow above.

See :ref:`ovphysx-live-ovd-recording` for a runnable headless example, destination options, and the
recording lifetime.

Inspect an OVPhysX capture
~~~~~~~~~~~~~~~~~~~~~~~~~~

A live capture is a physics recording for inspection. Use an external reader to examine the file.
With the PhysX SDK's ``ovd-inspect`` tool installed, summarize the recording to find its scenes and
completed frames.

.. code-block:: bash

   ovd-inspect summary captures/late.ovd

The `SDK inspection guide <https://nvidia-omniverse.github.io/PhysX/ovphysx/0.8.0/tutorials/omnipvd_recording.html#headless-capture-and-investigation>`_
shows how to inspect an object's recorded values. For visual playback in a compatible Kit application,
enable the OmniPVD extension and follow its `OVD import instructions <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/ux/source/omni.physx.pvd/docs/dev_guide/physx_visual_debugger.html#loading-an-ovd-file-from-the-file-import-menu>`_.


.. _Stage Recorder: https://docs.omniverse.nvidia.com/extensions/latest/ext_animation_stage-recorder.html
.. _Fabric: https://docs.omniverse.nvidia.com/kit/docs/usdrt.scenegraph/latest/usd_fabric_usdrt.html
.. _Omniverse Launcher: https://docs.omniverse.nvidia.com/launcher/latest/index.html
.. _tutorial on layering in Omniverse: https://www.youtube.com/watch?v=LTwmNkSDh-c&ab_channel=NVIDIAOmniverse
