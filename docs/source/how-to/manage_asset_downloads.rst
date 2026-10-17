:orphan:

.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _manage-asset-downloads:

Manage Asset Downloads
======================

Isaac Lab assets are hosted on AWS S3 and are downloaded as a workflow needs them. Use this guide
to cache frequently used assets, migrate an existing Nucleus setup, or select the regional asset
service for mainland China.


Cache remote assets
-------------------

Enable Hub Workstation Cache when repeated downloads are slow or the workstation has intermittent
network access.

Launch Isaac Sim:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         uv run --extra isaacsim isaaclab -s

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         uv run --extra isaacsim isaaclab -s

Select the ``CACHE:`` message in the upper-right corner and enable `Hub Workstation Cache
<https://docs.omniverse.nvidia.com/utilities/latest/cache/hub-workstation.html>`__. The first load
still downloads each asset; later runs use the local cache.

.. figure:: /source/_static/setup/asset_caching.jpg
   :align: center
   :figwidth: 100%
   :alt: Isaac Sim cache status message.

.. dropdown:: Detailed asset caching and Nucleus migration notes

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

            export ISAACSIM_ASSET_ROOT=/path/to/Assets/Isaac/X.Y

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            set ISAACSIM_ASSET_ROOT=C:\path\to\Assets\Isaac\X.Y

   The variable must point at the versioned asset root, that is the directory containing ``Isaac`` and
   ``NVIDIA``. It takes precedence over the value shipped in the experience files, and it always applies to
   the Isaac Lab asset paths such as :attr:`~isaaclab.utils.assets.ISAAC_NUCLEUS_DIR`. The Isaac Sim asset
   root resolves to the same value when you use one of the provided Isaac Lab experiences, because those
   load ``isaacsim.storage.native``. A custom experience that does not build on one of them may leave the
   Isaac Sim asset root pointing at the cloud.

   Isaac Lab supports the ``ISAACSIM_ASSET_ROOT`` and ``ISAACSIM_ASSET_REGION_PROFILE`` environment variables
   for runtime asset selection. It resolves asset paths once, when :mod:`isaaclab.utils.assets` is imported, so setting
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

Omniverse Nucleus and Omniverse Launcher are deprecated starting with Isaac Sim 4.5. Existing local
Nucleus installations continue to work.


.. _installation-asset-region-profiles:

Select an asset region
----------------------

An Asset Region Profile selects a compatible asset root and configures any storage settings required
for that service. Isaac Lab provides these profiles:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Profile
     - Use
   * - ``us``
     - Primary public asset service and explicit switchback profile.
   * - ``china``
     - Regional asset service for users in mainland China.

Set the profile before launching Isaac Lab. Clear ``ISAACSIM_ASSET_ROOT`` first because an explicit
asset-root override takes precedence over the selected profile.

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         unset ISAACSIM_ASSET_ROOT
         export ISAACSIM_ASSET_REGION_PROFILE=china

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         set ISAACSIM_ASSET_ROOT=
         set ISAACSIM_ASSET_REGION_PROFILE=china

Isaac Lab launchers and asset helpers apply the profile automatically. The same variable also selects
the profile when Isaac Lab launches Isaac Sim. In kit-less mode, Isaac Lab configures the required
``omni.client`` routing without requiring Isaac Sim.

A standalone kit-less script that calls ``omni.client`` before launching an Isaac Lab runtime must
initialize the profile first:

.. code-block:: python

   from isaaclab.utils.assets import configure_asset_region_profile

   configure_asset_region_profile()

To return to the primary service, clear ``ISAACSIM_ASSET_ROOT`` and select the ``us`` profile.

The latest published China mirror inventory is the
`6.1 asset availability manifest <https://assets.simready.cn/manifests/isaac/6.1/asset-availability.csv>`__.
The ``isaac_version`` field identifies the asset release. Each ``asset_path`` is the full path to a
file relative to the versioned asset root's ``Isaac`` directory. A ``status`` value of ``available``
reports that the file is mirrored, ``reason_code`` explains other statuses when provided, and
``checked_at`` records when the status last changed. A path with no row is not mirrored. The manifest
does not confirm the availability of paths outside the root's ``Isaac`` directory.

Build paths from profile-resolved constants such as
:attr:`~isaaclab.utils.assets.ISAAC_NUCLEUS_DIR` and
:attr:`~isaaclab.utils.assets.ISAACLAB_NUCLEUS_DIR`. Do not hardcode the profile's storage endpoint or
derive direct object URLs from the manifest. Opening an object-storage URL directly in a browser or
with ``curl`` can return HTTP 403 because it bypasses the profile's CDN routing.
