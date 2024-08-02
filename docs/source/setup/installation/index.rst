Installation Guide
===================

.. image:: https://img.shields.io/badge/IsaacSim-4.1.0-silver.svg
   :target: https://developer.nvidia.com/isaac-sim
   :alt: IsaacSim 4.1.0

.. image:: https://img.shields.io/badge/python-3.10-blue.svg
   :target: https://www.python.org/downloads/release/python-31013/
   :alt: Python 3.10

.. image:: https://img.shields.io/badge/platform-linux--64-orange.svg
   :target: https://releases.ubuntu.com/20.04/
   :alt: Ubuntu 20.04

.. image:: https://img.shields.io/badge/platform-windows--64-orange.svg
   :target: https://www.microsoft.com/en-ca/windows/windows-11
   :alt: Windows 11

.. caution::

   We have dropped support for Isaac Sim versions 2023.1.1 and below. We recommend using the latest
   Isaac Sim 4.1.0 release to benefit from the latest features and improvements.

   For more information, please refer to the
   `Isaac Sim release notes <https://docs.omniverse.nvidia.com/isaacsim/latest/release_notes.html>`__.

.. note::

    We recommend system requirements with at least 32GB RAM and 16GB VRAM for Isaac Lab.
    For the full list of system requirements for Isaac Sim, please refer to the
    `Isaac Sim system requirements <https://docs.omniverse.nvidia.com/isaacsim/latest/installation/requirements.html#system-requirements>`_.

As an experimental feature since Isaac Sim 4.0 release, Isaac Sim can also be installed through pip.
This simplifies the installation
process by avoiding the need to download the Omniverse Launcher and installing Isaac Sim through
the launcher. Therefore, there are two ways to install Isaac Lab:

.. toctree::
    :maxdepth: 2

    Installation using Isaac Sim pip (experimental) <pip_installation>
    binaries_installation
    verifying_installation
    cloud_installation
