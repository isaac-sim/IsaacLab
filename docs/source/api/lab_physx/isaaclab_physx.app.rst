isaaclab\_physx.app
===================

.. automodule:: isaaclab_physx.app

   .. rubric:: Classes

   .. autosummary::

      KitLauncher

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
         uv run --extra isaacsim isaaclab demo zoo --physics isaacsim_physx --viz kit

      Alternatively, set the environment variable inline for a single invocation:

      .. code-block:: bash

         LIVESTREAM=2 uv run --extra isaacsim isaaclab demo zoo --physics isaacsim_physx --viz kit

   .. tab-item:: :icon:`fa-brands fa-linux` Linux aarch64 (DGX Spark)
      :sync: linux-aarch64

      .. code-block:: bash

         export LIVESTREAM=2
         # run the python script
         LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 uv run --extra isaacsim isaaclab demo zoo --physics isaacsim_physx --viz kit

      Alternatively, set the environment variable inline for a single invocation:

      .. code-block:: bash

         LIVESTREAM=2 LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 uv run --extra isaacsim isaaclab demo zoo --physics isaacsim_physx --viz kit

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

      In Command Prompt:

      .. code-block:: batch

         set LIVESTREAM=2
         uv run --extra isaacsim isaaclab demo zoo --physics isaacsim_physx --viz kit

      In PowerShell:

      .. code-block:: powershell

         $env:LIVESTREAM = "2"
         uv run --extra isaacsim isaaclab demo zoo --physics isaacsim_physx --viz kit

      .. note::

         The POSIX inline ``VAR=value <command>`` prefix form used on Linux (for example
         ``LIVESTREAM=2 uv run ...``) has no Windows equivalent; use one of the two forms
         above instead.


Overriding the environment variables
------------------------------------

Scripts do not construct the :class:`KitLauncher` themselves. :func:`~isaaclab.app.launch_simulation`
constructs it when the resolved configuration needs Isaac Sim / Kit, and passes it the launcher arguments.
Arguments that are not at their default values override the environment variables. They can be passed as
an :class:`argparse.Namespace` or as a dictionary:

.. code:: python

   import argparse

   from isaaclab.app import add_launcher_args, launch_simulation

   parser = argparse.ArgumentParser()
   # add your own arguments
   # ....
   # add the launcher arguments for the command line
   add_launcher_args(parser)
   args = parser.parse_args()

   # -- Option 1: pass the parsed arguments
   with launch_simulation(env_cfg, args):
       ...
   # -- Option 2: pass the settings as a dictionary
   with launch_simulation(env_cfg, {"livestream": 2, "enable_cameras": True}):
       ...


Kit Launcher
------------

.. autoclass:: KitLauncher
   :members:
   :show-inheritance:

.. _livestream: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html
.. _`WebRTC Livestream`: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html#isaac-sim-short-webrtc-streaming-client
