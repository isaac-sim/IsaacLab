Recording Animations of Simulations
===================================

.. currentmodule:: isaaclab

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


In Isaac Lab, we directly use the `Stage Recorder`_ extension to record the animation of the physics simulation.
This is available as a feature in the :class:`~isaaclab.envs.ui.BaseEnvWindow` class.
However, to record the animation of a simulation, you need to disable `Fabric`_ to allow reading and writing
all the changes (such as motion and USD properties) to the USD stage.


Stage Recorder Settings
~~~~~~~~~~~~~~~~~~~~~~~

Isaac Lab integration of the `Stage Recorder`_ extension assumes certain default settings. If you want to change the
settings, you can directly use the `Stage Recorder`_ extension in the Omniverse Create application.

.. dropdown:: Settings used in base_env_window.py
  :icon: code

  .. literalinclude:: ../../../source/isaaclab/isaaclab/envs/ui/base_env_window.py
    :language: python
    :linenos:
    :pyobject: BaseEnvWindow._toggle_recording_animation_fn


Example Usage
~~~~~~~~~~~~~

In all environment standalone scripts, Fabric can be disabled by passing the ``--disable_fabric`` flag to the script.
Here we run the state-machine example and record the animation of the simulation.

.. code-block:: bash

  ./isaaclab.sh -p scripts/environments/state_machine/lift_cube_sm.py --num_envs 8 --device cpu --disable_fabric


On running the script, the Isaac Lab UI window opens with the button "Record Animation" in the toolbar.
Clicking this button starts recording the animation of the simulation. On clicking the button again, the
recording stops. The recorded animation and the original stage (with all physics disabled) are saved
to the ``recordings`` folder in the current working directory. The files are stored in the ``usd`` format:

- ``Stage.usd``: The original stage with all physics disabled
- ``TimeSample_tk001.usd``: The timeSampled file containing the recorded animation

You can open Omniverse Isaac Sim application to play back the animation. There are many ways to launch
the application (such as from terminal or `Omniverse Launcher`_). Here we use the terminal to open the
application and play the animation.

.. code-block:: bash

  ./isaaclab.sh -s  # Opens Isaac Sim application through _isaac_sim/isaac-sim.sh

On a new stage, add the ``Stage.usd`` as a sublayer and then add the ``TimeSample_tk001.usd`` as a sublayer.
You can do this by dragging and dropping the files from the file explorer to the stage. Please check out
the `tutorial on layering in Omniverse`_ for more details.

You can then play the animation by pressing the play button.

.. _Stage Recorder: https://docs.omniverse.nvidia.com/extensions/latest/ext_animation_stage-recorder.html
.. _Fabric: https://docs.omniverse.nvidia.com/kit/docs/usdrt/latest/docs/usd_fabric_usdrt.html
.. _Omniverse Launcher: https://docs.omniverse.nvidia.com/launcher/latest/index.html
.. _tutorial on layering in Omniverse: https://www.youtube.com/watch?v=LTwmNkSDh-c&ab_channel=NVIDIAOmniverse
