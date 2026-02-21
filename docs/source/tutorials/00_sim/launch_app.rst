Deep-dive into AppLauncher
==========================

.. currentmodule:: isaaclab

In this tutorial, we will dive into the :class:`app.AppLauncher` class to configure the simulator using
CLI arguments and environment variables (envars). Particularly, we will demonstrate how to use
:class:`~app.AppLauncher` to enable livestreaming and configure the :class:`isaacsim.simulation_app.SimulationApp`
instance it wraps, while also allowing user-provided options.

The :class:`~app.AppLauncher` is a wrapper for :class:`~isaacsim.simulation_app.SimulationApp` to simplify
its configuration. The :class:`~isaacsim.simulation_app.SimulationApp` has many extensions that must be
loaded to enable different capabilities, and some of these extensions are order- and inter-dependent.
Additionally, there are startup options such as ``headless`` which must be set at instantiation time,
and which have an implied relationship with some extensions, e.g. the livestreaming extensions.
The :class:`~app.AppLauncher` presents an interface that can handle these extensions and startup
options in a portable manner across a variety of use cases. To achieve this, we offer CLI and envar
flags which can be merged with user-defined CLI args, while passing forward arguments intended
for :class:`~isaacsim.simulation_app.SimulationApp`.


The Code
--------

The tutorial corresponds to the ``launch_app.py`` script in the
``scripts/tutorials/00_sim`` directory.

.. dropdown:: Code for launch_app.py
   :icon: code

   .. literalinclude:: ../../../../scripts/tutorials/00_sim/launch_app.py
      :language: python
      :emphasize-lines: 18-40
      :linenos:

The Code Explained
------------------

Adding arguments to the argparser
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~app.AppLauncher` is designed to be compatible with custom CLI args that users need for
their own scripts, while still providing a portable CLI interface.

In this tutorial, a standard :class:`argparse.ArgumentParser` is instantiated and given the
script-specific ``--size`` argument, as well as the arguments ``--height`` and ``--width``.
The latter are ingested by :class:`~isaacsim.simulation_app.SimulationApp`.

The argument ``--size`` is not used by :class:`~app.AppLauncher`, but will merge seamlessly
with the :class:`~app.AppLauncher` interface. In-script arguments can be merged with the
:class:`~app.AppLauncher` interface via the :meth:`~app.AppLauncher.add_app_launcher_args` method,
which will return a modified :class:`~argparse.ArgumentParser` with the :class:`~app.AppLauncher`
arguments appended. This can then be processed into an :class:`argparse.Namespace` using the
standard :meth:`argparse.ArgumentParser.parse_args` method and passed directly to
:class:`~app.AppLauncher` for instantiation.

.. literalinclude::  ../../../../scripts/tutorials/00_sim/launch_app.py
   :language: python
   :start-at: import argparse
   :end-at: simulation_app = app_launcher.app

The above only illustrates only one of several ways of passing arguments to :class:`~app.AppLauncher`.
Please consult its documentation page to see further options.

Understanding the output of --help
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While executing the script, we can pass the ``--help`` argument and see the combined outputs of the
custom arguments and those from :class:`~app.AppLauncher`.

.. code-block:: console

   ./isaaclab.sh -p scripts/tutorials/00_sim/launch_app.py --help

   [INFO] Using python from: /isaac-sim/python.sh
   [INFO][AppLauncher]: The argument 'width' will be used to configure the SimulationApp.
   [INFO][AppLauncher]: The argument 'height' will be used to configure the SimulationApp.
   usage: launch_app.py [-h] [--size SIZE] [--width WIDTH] [--height HEIGHT] [--headless] [--livestream {0,1,2}]
                        [--enable_cameras] [--verbose] [--experience EXPERIENCE]

   Tutorial on running IsaacSim via the AppLauncher.

   options:
   -h, --help            show this help message and exit
   --size SIZE           Side-length of cuboid
   --width WIDTH         Width of the viewport and generated images. Defaults to 1280
   --height HEIGHT       Height of the viewport and generated images. Defaults to 720

   app_launcher arguments:
   --headless            Force display off at all times.
   --livestream {0,1,2}
                         Force enable livestreaming. Mapping corresponds to that for the "LIVESTREAM" environment variable.
   --enable_cameras      Enable cameras when running without a GUI.
   --verbose             Enable verbose terminal logging from the SimulationApp.
   --experience EXPERIENCE
                         The experience file to load when launching the SimulationApp.

                         * If an empty string is provided, the experience file is determined based on the headless flag.
                         * If a relative path is provided, it is resolved relative to the `apps` folder in Isaac Sim and
                           Isaac Lab (in that order).

This readout details the ``--size``, ``--height``, and ``--width`` arguments defined in the script directly,
as well as the :class:`~app.AppLauncher` arguments.

The ``[INFO]`` messages preceding the help output also reads out which of these arguments are going
to be interpreted as arguments to the :class:`~isaacsim.simulation_app.SimulationApp` instance which the
:class:`~app.AppLauncher` class wraps. In this case, it is ``--height`` and ``--width``. These
are classified as such because they match the name and type of an argument which can be processed
by :class:`~isaacsim.simulation_app.SimulationApp`. Please refer to the `specification`_ for such arguments
for more examples.

Using environment variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^

As noted in the help message, the :class:`~app.AppLauncher` arguments (``--livestream``, ``--headless``)
have corresponding environment variables (envar) as well. These are detailed in :mod:`isaaclab.app`
documentation. Providing any of these arguments through CLI is equivalent to running the script in a shell
environment where the corresponding envar is set.

The support for :class:`~app.AppLauncher` envars are simply a convenience to provide session-persistent
configurations, and can be set in the user's ``${HOME}/.bashrc`` for persistent settings between sessions.
In the case where these arguments are provided from the CLI, they will override their corresponding envar,
as we will demonstrate later in this tutorial.

These arguments can be used with any script that starts the simulation using :class:`~app.AppLauncher`,
with one exception, ``--enable_cameras``. This setting sets the rendering pipeline to use the
offscreen renderer. However, this setting is only compatible with the :class:`isaaclab.sim.SimulationContext`.
It will not work with Isaac Sim's :class:`isaacsim.core.api.simulation_context.SimulationContext` class.
For more information on this flag, please see the :class:`~app.AppLauncher` API documentation.


The Code Execution
------------------

We will now run the example script:

.. code-block:: console

   LIVESTREAM=2 ./isaaclab.sh -p scripts/tutorials/00_sim/launch_app.py --size 0.5

This will spawn a 0.5m\ :sup:`3` volume cuboid in the simulation. No GUI will appear, equivalent
to if we had passed the ``--headless`` flag because headlessness is implied by our ``LIVESTREAM``
envar. If a visualization is desired, we could get one via Isaac's `WebRTC Livestreaming`_. Streaming
is currently the only supported method of visualization from within the container. The
process can be killed by pressing ``Ctrl+C`` in the launching terminal.

.. figure:: ../../_static/tutorials/tutorial_launch_app.jpg
    :align: center
    :figwidth: 100%
    :alt: result of launch_app.py

Now, let's look at how :class:`~app.AppLauncher` handles conflicting commands:

.. code-block:: console

   LIVESTREAM=0 ./isaaclab.sh -p scripts/tutorials/00_sim/launch_app.py --size 0.5 --livestream 2

This will cause the same behavior as in the previous run, because although we have set ``LIVESTREAM=0``
in our envars, CLI args such as ``--livestream`` take precedence in determining behavior. The process can
be killed by pressing ``Ctrl+C`` in the launching terminal.

Finally, we will examine passing arguments to :class:`~isaacsim.simulation_app.SimulationApp` through
:class:`~app.AppLauncher`:

.. code-block:: console

   LIVESTREAM=2 ./isaaclab.sh -p scripts/tutorials/00_sim/launch_app.py --size 0.5 --width 1920 --height 1080

This will cause the same behavior as before, but now the viewport will be rendered at 1920x1080p resolution.
This can be useful when we want to gather high-resolution video, or we can specify a lower resolution if we
want our simulation to be more performant. The process can be killed by pressing ``Ctrl+C`` in the launching
terminal.


.. _specification: https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.simulation_app/docs/index.html#isaacsim.simulation_app.SimulationApp.DEFAULT_LAUNCHER_CONFIG
.. _WebRTC Livestreaming: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html#isaac-sim-short-webrtc-streaming-client
