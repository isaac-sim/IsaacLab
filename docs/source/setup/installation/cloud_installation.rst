Cloud Deployment
================

Isaac Lab can be run in various cloud infrastructures with the use of
`Isaac Automator <https://github.com/isaac-sim/IsaacAutomator>`__ (v4).

Isaac Automator allows quick deployment of Isaac Sim, Isaac Lab, and Isaac Lab Arena
onto public clouds (AWS, GCP, Azure, and Alibaba Cloud are currently supported).
The result is a fully configured remote desktop cloud workstation (Isaac Workstation),
which can be used for development and testing of Isaac Lab within minutes and on a budget.
Isaac Automator supports a variety of GPU instances and stop/start functionality
to save on cloud costs, and provides tools to aid the workflow
(uploading and downloading data, autorun scripts, deployment management, etc.).


System Requirements
-------------------

Isaac Automator requires having ``docker`` pre-installed on the system.

* To install Docker, please follow the instructions for your operating system on the
  `Docker website`_.
* Follow the post-installation steps for Docker on the `post-installation steps`_ page.
  These steps allow you to run Docker without using ``sudo``.


Installing Isaac Automator
---------------------------

For the most up-to-date and complete installation instructions, please refer to
the `Isaac Automator README <https://github.com/isaac-sim/IsaacAutomator?tab=readme-ov-file#installation>`__.

To use Isaac Automator, first clone the repo:

.. tab-set::

   .. tab-item:: HTTPS

      .. code-block:: bash

         git clone https://github.com/isaac-sim/IsaacAutomator.git

   .. tab-item:: SSH

      .. code-block:: bash

         git clone git@github.com:isaac-sim/IsaacAutomator.git


Building the Container
----------------------

Build the Isaac Automator container:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux / macOS
      :sync: linux

      .. code-block:: bash

         ./build

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         docker build --platform linux/x86_64 -t isaac_automator .

This will build the Isaac Automator container and tag it as ``isaac_automator``.


Deploying an Isaac Workstation
------------------------------

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux / macOS
      :sync: linux

      Enter the Automator container and run the deployment command:

      .. code-block:: bash

         ./run
         # inside container:
         ./deploy-aws

      Alternatively, run it in one step:

      .. code-block:: bash

         ./run ./deploy-aws

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         docker run --platform linux/x86_64 -it --rm -v .:/app isaac_automator bash
         :: inside container:
         ./deploy-aws

Replace ``deploy-aws`` with ``deploy-gcp``, ``deploy-azure``, or ``deploy-alicloud``
for other cloud providers.

.. note::

   The ``--isaaclab`` and ``--isaacsim`` flags accept any valid Git reference
   to specify the version to deploy. Use ``--isaaclab no`` or ``--isaacsim no``
   to skip installation of the respective component.

   .. code-block:: bash

      ./deploy-aws --isaaclab v3.0.0 --isaacsim main

On the first run (or when credentials expire), you will be prompted to enter
your cloud credentials. Credentials are stored in ``state/`` and persist
across container restarts. Run ``./deploy-<cloud> --help`` to see all available
options.

Key deployment options:

- ``--instance-type`` -- Cloud VM instance type.
- ``--isaacsim`` / ``--isaaclab`` / ``--isaaclab-arena`` -- Git ref for the version
  to install, or ``no`` to skip.
- ``--existing`` -- What to do if a deployment already exists: ``ask`` (default),
  ``repair``, ``modify``, ``replace``, or ``run_ansible``.
- ``--from-image`` -- Deploy from a pre-built VM image for faster provisioning
  (AWS only at this time).

Connecting to the Isaac Workstation
-----------------------------------

Deployed Isaac Workstations can be accessed via:

- **SSH**: ``./ssh <deployment-name>``
- **noVNC** (browser-based): ``./novnc <deployment-name>``
- **NoMachine** (remote desktop client)

Connection instructions are displayed at the end of the deployment command
output and saved in ``state/<deployment-name>/info.txt``.


Running Isaac Lab on the Cloud
------------------------------

Isaac Lab is installed from source on the deployed workstation at ``~/IsaacLab``.
To run Isaac Lab commands, open a terminal on the workstation:

.. code-block:: bash

   ~/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
     --task=Isaac-Cartpole-Direct-v0 --headless


Pausing and Resuming
--------------------

You can stop and restart instances to save on cloud costs:

.. code-block:: bash

   # inside the Automator container:
   ./stop <deployment-name>
   ./start <deployment-name>

Use ``./start <deployment-name> --quick`` to skip full Ansible provisioning
and only run the autorun script.


Uploading and Downloading Data
------------------------------

.. code-block:: bash

   # upload local uploads/ folder to the instance
   ./upload <deployment-name>

   # download results from the instance to local results/ folder
   ./download <deployment-name>


Destroying a Deployment
-----------------------

To save costs, destroy deployments when no longer needed:

.. code-block:: bash

   # inside the Automator container:
   ./destroy <deployment-name>

.. note::

   Deployment metadata is stored in the ``state/`` directory. Do not delete this
   directory, as it is required for managing deployments.


.. _`Docker website`: https://docs.docker.com/engine/install/
.. _`post-installation steps`: https://docs.docker.com/engine/install/linux-postinstall/
