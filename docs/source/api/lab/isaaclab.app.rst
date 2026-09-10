isaaclab.app
============

.. automodule:: isaaclab.app

   .. rubric:: Classes

   .. autosummary::

      AppLauncher
      LoadingScreen
      Scan

   .. rubric:: Functions

   .. autosummary::

      launch_simulation
      make_physics_cfg
      report_activity
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

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux x86_64
      :sync: linux-x86_64

      .. code-block:: bash

         export LIVESTREAM=2
         # run the python script
         uv run --extra isaacsim python scripts/demos/quadrupeds.py

      Alternatively, set the environment variable inline for a single invocation:

      .. code-block:: bash

         LIVESTREAM=2 uv run --extra isaacsim python scripts/demos/quadrupeds.py

   .. tab-item:: :icon:`fa-brands fa-linux` Linux aarch64 (DGX Spark)
      :sync: linux-aarch64

      .. code-block:: bash

         export LIVESTREAM=2
         # run the python script
         LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 uv run --extra isaacsim python scripts/demos/quadrupeds.py

      Alternatively, set the environment variable inline for a single invocation:

      .. code-block:: bash

         LIVESTREAM=2 LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 uv run --extra isaacsim python scripts/demos/quadrupeds.py

      .. note::

         Direct Python commands that import Isaac Sim on aarch64 require the
         ``LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1`` prefix shown above. See
         :ref:`installation-method-uv`.

      .. warning::

         Livestreaming is not currently supported or validated on DGX Spark. See the
         :doc:`/source/setup/installation/index` for the current list of features not
         yet validated on this platform.

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      In an elevated Command Prompt:

      .. code-block:: batch

         set LIVESTREAM=2
         uv run --extra isaacsim python scripts/demos/quadrupeds.py

      In PowerShell:

      .. code-block:: powershell

         $env:LIVESTREAM = "2"
         uv run --extra isaacsim python scripts/demos/quadrupeds.py

      .. note::

         The POSIX inline ``VAR=value <command>`` prefix form used on Linux (for example
         ``LIVESTREAM=2 uv run ...``) has no Windows equivalent; use one of the two forms
         above instead.


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

Additional Public Classes
-------------------------

The following classes are part of the public :mod:`isaaclab.app` API.

.. currentmodule:: isaaclab.app

.. autosummary::
   :nosignatures:

   LoadingScreen
   SettingsManager

.. autoclass:: LoadingScreen
   :show-inheritance:

.. autoclass:: SettingsManager
   :show-inheritance:
