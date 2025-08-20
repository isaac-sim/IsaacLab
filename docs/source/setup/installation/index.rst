.. _isaaclab-installation-root:

Local Installation
==================

.. image:: https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg
   :target: https://developer.nvidia.com/isaac-sim
   :alt: IsaacSim 5.0.0

.. image:: https://img.shields.io/badge/python-3.11-blue.svg
   :target: https://www.python.org/downloads/release/python-31013/
   :alt: Python 3.11

.. image:: https://img.shields.io/badge/platform-linux--64-orange.svg
   :target: https://releases.ubuntu.com/22.04/
   :alt: Ubuntu 22.04

.. image:: https://img.shields.io/badge/platform-windows--64-orange.svg
   :target: https://www.microsoft.com/en-ca/windows/windows-11
   :alt: Windows 11

.. caution::

   We have dropped support for Isaac Sim versions 4.2.0 and below. We recommend using the latest
   Isaac Sim 5.0.0 release to benefit from the latest features and improvements.

   For more information, please refer to the
   `Isaac Sim release notes <https://docs.isaacsim.omniverse.nvidia.com/latest/overview/release_notes.html#>`__.

.. note::

    We recommend system requirements with at least 32GB RAM and 16GB VRAM for Isaac Lab.
    For workflows with rendering enabled, additional VRAM may be required.
    For the full list of system requirements for Isaac Sim, please refer to the
    `Isaac Sim system requirements <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html#system-requirements>`_.

    For details on driver requirements, please see the `Technical Requirements <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/common/technical-requirements.html>`_ guide

    * See `Linux Troubleshooting <https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html>`_ to resolve driver installation issues in linux
    * If you are on a new GPU or are experiencing issues with the current drivers, we recommend installing the **latest production branch version** drivers from the `Unix Driver Archive <https://www.nvidia.com/en-us/drivers/unix/>`_ using the ``.run`` installer on Linux.
    * NVIDIA driver version ``535.216.01`` or later is recommended when upgrading to **Ubuntu 22.04.5 kernel 6.8.0-48-generic** or later


Isaac Lab is built on top of the Isaac Sim platform. Therefore, it is required to first install Isaac Sim
before using Isaac Lab.

Both Isaac Sim and Isaac Lab provide two ways of installation:
either through binary download/source file, or through Python's package installer ``pip``.

The method of installation may depend on the use case and the level of customization desired from users.
For example, installing Isaac Sim from pip will be a simpler process than installing it from binaries,
but the source code will then only be accessible through the installed source package and not through the direct binary download.

Similarly, installing Isaac Lab through pip is only recommended for workflows that use external launch scripts outside of Isaac Lab.
The Isaac Lab pip packages only provide the core framework extensions for Isaac Lab and does not include any of the
standalone training, inferencing, and example scripts. Therefore, this workflow is recommended for projects that are
built as external extensions outside of Isaac Lab, which utilizes user-defined runner scripts.

We recommend using Isaac Sim pip installation for a simplified installation experience.

For users getting started with Isaac Lab, we recommend installing Isaac Lab by cloning the repo.


.. toctree::
    :maxdepth: 2

    Pip installation (recommended) <pip_installation>
    Binary installation <binaries_installation>
    Advanced installation (Isaac Lab pip) <isaaclab_pip_installation>
    Asset caching <asset_caching>
