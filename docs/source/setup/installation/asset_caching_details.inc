.. rubric:: Asset Caching

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

.. rubric:: Using a Local Asset Root

Machines that are air-gapped or otherwise unable to reach the AWS S3 buckets can point Isaac Lab at a
local or self-hosted copy of the assets. Set the ``ISAACSIM_ASSET_ROOT`` environment variable before
launching:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         export ISAACSIM_ASSET_ROOT=/path/to/Assets/Isaac/6.0

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         set ISAACSIM_ASSET_ROOT=C:\path\to\Assets\Isaac\6.0

The variable must point at the versioned asset root, that is the directory containing ``Isaac`` and
``NVIDIA``. It takes precedence over the value shipped in the experience files, and it always applies to
the Isaac Lab asset paths such as :attr:`~isaaclab.utils.assets.ISAAC_NUCLEUS_DIR`. The Isaac Sim asset
root resolves to the same value when you use one of the provided Isaac Lab experiences, because those
load ``isaacsim.storage.native``. A custom experience that does not build on one of them may leave the
Isaac Sim asset root pointing at the cloud.

The environment variable is the only supported runtime override for the Isaac Lab asset paths. Isaac Lab
resolves them once, when :mod:`isaaclab.utils.assets` is imported, so setting
``persistent.isaac.asset_root.default`` on the command line or changing it at runtime does not move them,
even though Isaac Sim itself honors those forms. Editing ``persistent.isaac.asset_root.default`` in the
experience file does work, because that is the value Isaac Lab reads when the variable is unset.

.. note::
   If you previously redirected Isaac Lab assets by editing ``persistent.isaac.asset_root.cloud`` in an
   experience file, that edit no longer has any effect. Isaac Lab now reads
   ``persistent.isaac.asset_root.default`` first, and the shipped experiences leave it pointing at the
   cloud. Set ``ISAACSIM_ASSET_ROOT`` instead, or edit ``persistent.isaac.asset_root.default``.

For instructions on downloading the asset pack, see the
`Isaac Sim asset documentation <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_faq.html#isaac-sim-setup-assets-content-pack>`__.

.. rubric:: Nucleus


Before Isaac Sim 4.5, assets were accessed via the Omniverse Nucleus server, including setups with local Nucleus instances.

.. warning::
   Starting with Isaac Sim 4.5, the Omniverse Nucleus server and Omniverse Launcher are deprecated.
   Existing Nucleus setups will continue to work, so if you have a local Nucleus server already configured,
   you may continue to use it.
