isaaclab.app
============

.. automodule:: isaaclab.app

   .. rubric:: Classes

   .. autosummary::

      AppLauncher
      Scan

   .. rubric:: Functions

   .. autosummary::

      launch_simulation
      make_physics_cfg
      scan


Environment variables
---------------------

The following details the behavior of the class based on the environment variables:

* **Headless mode**: If the environment variable ``HEADLESS=1``, then SimulationApp will be started in headless mode.
  If ``LIVESTREAM={1,2}``, then it will supersede the ``HEADLESS`` envvar and force headlessness.

  * ``HEADLESS=1`` causes the app to run in headless mode.

* **Livestreaming**: If the environment variable ``LIVESTREAM={1,2}`` , then `livestream`_ is enabled. Any
  of the livestream modes being true forces the app to run in headless mode.

  * ``LIVESTREAM=1`` enables streaming via the `WebRTC Livestream`_ extension over **public networks**. This allows users to
    connect through the WebRTC Client using the WebRTC protocol.
  * ``LIVESTREAM=2`` enables streaming via the `WebRTC Livestream`_ extension over **private and local networks**. This allows users to
    connect through the WebRTC Client using the WebRTC protocol.

  .. note::

    Each Isaac Sim instance can only connect to one streaming client.
    Connecting to an Isaac Sim instance that is currently serving a streaming client
    results in an error for the second user.

* **Public IP Address**: When using the environment variable ``LIVESTREAM={1,2}``, set the ``PUBLIC_IP`` envvar to define the public IP address endpoint for livestreaming remotely.

Camera and offscreen rendering support is enabled automatically. No environment variable or command-line
option is required for camera tasks.


To set the environment variables, one can use the following command in the terminal:

.. code:: bash

   export LIVESTREAM=2
   # run the python script
   uv run --extra isaacsim python scripts/demos/quadrupeds.py

Alternatively, one can set the environment variables to the python script directly:

.. code:: bash

   LIVESTREAM=2 uv run --extra isaacsim python scripts/demos/quadrupeds.py


Overriding the environment variables
------------------------------------

The environment variables can be overridden in the python script itself using the :class:`AppLauncher`.
These can be passed as a dictionary, a :class:`argparse.Namespace` object or as keyword arguments.
When the passed arguments are not the default values, then they override the environment variables.

The following snippet shows how use the :class:`AppLauncher` in different ways:

.. code:: python

   import argparse

   from isaaclab.app import AppLauncher

   # add argparse arguments
   parser = argparse.ArgumentParser()
   # add your own arguments
   # ....
   # add app launcher arguments for cli
   AppLauncher.add_app_launcher_args(parser)
   # parse arguments
   args = parser.parse_args()

   # launch omniverse isaac-sim app
   # -- Option 1: Pass the settings as a Namespace object
   app_launcher = AppLauncher(args).app
   # -- Option 2: Pass the settings as keywords arguments
   app_launcher = AppLauncher(headless=args.headless, livestream=args.livestream)
   # -- Option 3: Pass the settings as a dictionary
   app_launcher = AppLauncher(vars(args))
   # -- Option 4: Pass no settings
   app_launcher = AppLauncher()

   # obtain the launched app
   simulation_app = app_launcher.app


Simulation App Launcher
-----------------------

.. autoclass:: AppLauncher
   :members:


Simulation Launcher
-------------------

.. autofunction:: launch_simulation

.. autofunction:: make_physics_cfg

.. autofunction:: scan

.. autoclass:: Scan
   :members:


.. _livestream: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html
.. _`WebRTC Livestream`: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html#isaac-sim-short-webrtc-streaming-client
