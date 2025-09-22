Asset Caching
=============

Assets used in Isaac Lab are hosted on AWS S3 buckets on the cloud.
Asset loading time can depend on your network connection and geographical location.
In some cases, it is possible that asset loading times can be long when assets are pulled from the AWS servers.

If you run into cases where assets take a few minutes to load for each run,
we recommend enabling asset caching following the below steps.

First, launch the Isaac Sim application:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         ./isaaclab.sh -s

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         isaaclab.bat -s

On the top right of the Isaac Lab or Isaac Sim app, look for the icon labeled ``CACHE:``.
You may see a message such as ``HUB NOT DETECTED`` or ``NEW VERSION DETECTED``.

Click the message to enable `Hub <https://docs.omniverse.nvidia.com/utilities/latest/cache/hub-workstation.html>`_.
Hub automatically manages local caching for Isaac Lab assets, so subsequent runs will use cached files instead of
downloading from AWS each time.

.. figure:: /source/_static/setup/asset_caching.jpg
    :align: center
    :figwidth: 100%
    :alt: Simulator with cache messaging.

Hub provides better control and management of cached assets, making workflows faster and more reliable, especially
in environments with limited or intermittent internet access.

.. note::
   The first time you run Isaac Lab, assets will still need to be pulled from the cloud, which could lead
   to longer loading times.  Once cached, loading times will be significantly reduced on subsequent runs.

Nucleus
-------


Before Isaac Sim 4.5, assets were accessed via the Omniverse Nucleus server, including setups with local Nucleus instances.

.. warning::
   Starting with Isaac Sim 4.5, the Omniverse Nucleus server and Omniverse Launcher are deprecated.
   Existing Nucleus setups will continue to work, so if you have a local Nucleus server already configured,
   you may continue to use it.
