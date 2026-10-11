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

   .. include:: ../setup/installation/asset_caching_details.inc

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

The ``china`` profile publishes an
`asset availability manifest <https://assets.simready.cn/manifests/isaac/6.1/asset-availability.csv>`__.
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
