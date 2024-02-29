Recording Animations of Simulations
===================================

.. currentmodule:: omni.isaac.orbit

Omniverse includes tools to record animations of physics simulations. The `Stage Recorder`_ extension
listens to all the motion and USD property changes within a USD stage and records them to a USD file.
This file contains the time samples of the changes, which can be played back to render the animation.

The timeSampled USD file only contains the changes to the stage. It uses the same hierarchy as the original
stage at the time of recording. This allows adding the animation to the original stage, or to a different
stage with the same hierarchy. The timeSampled file can be directly added as a sublayer to the original stage
to play back the animation.

.. note::

  Omniverse only supports playing animation or playing physics on a USD prim at the same time. If you want to
  play back the animation of a USD prim, you need to disable the physics simulation on the prim.


In Orbit, we directly use the `Stage Recorder`_ extension to record the animation of the physics simulation.
This is available as a feature in the :class:`~omni.isaac.orbit.envs.ui.BaseEnvWindow` class.
However, to record the animation of a simulation, you need to disable `Fabric`_ to allow reading and writing
all the changes (such as motion and USD properties) to the USD stage.


Stage Recorder Settings
~~~~~~~~~~~~~~~~~~~~~~~

Orbit integration of the `Stage Recorder`_ extension assumes certain default settings. If you want to change the
settings, you can directly use the `Stage Recorder`_ extension in the Omniverse Create application.

.. dropdown:: Settings used in base_env_window.py
  :icon: code

  .. literalinclude:: ../../../source/extensions/omni.isaac.orbit/omni/isaac/orbit/envs/ui/base_env_window.py
    :language: python
    :linenos:
    :pyobject: BaseEnvWindow._toggle_recording_animation_fn


Example Usage
~~~~~~~~~~~~~

In all environment standalone scripts, Fabric can be disabled by passing the ``--disable_fabric`` flag to the script.
Here we run the state-machine example and record the animation of the simulation.

.. code-block:: bash

  ./orbit.sh -p source/standalone/environments/state_machine/lift_cube_sm.py --num_envs 8 --cpu --disable_fabric


On running the script, the Orbit UI window opens with the button "Record Animation" in the toolbar.
Clicking this button starts recording the animation of the simulation. On clicking the button again, the
recording stops. The recorded animation and the original stage (with all physics disabled) are saved
to the ``recordings`` folder in the current working directory. The files are stored in the ``usd`` format:

- ``Stage.usd``: The original stage with all physics disabled
- ``TimeSample_tk001.usd``: The timeSampled file containing the recorded animation

You can open Omniverse Create application to play back the animation. On a new stage, add the ``Stage.usd``
as a sublayer and then add the ``TimeSample_tk001.usd`` as a sublayer. You can then play the animation by
pressing the play button.

.. _Stage Recorder: https://docs.omniverse.nvidia.com/extensions/latest/ext_animation_stage-recorder.html
.. _Fabric: https://docs.omniverse.nvidia.com/kit/docs/usdrt/latest/docs/usd_fabric_usdrt.html
