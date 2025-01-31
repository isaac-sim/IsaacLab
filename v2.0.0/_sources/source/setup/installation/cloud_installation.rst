Running Isaac Lab in the Cloud
==============================

Isaac Lab can be run in various cloud infrastructures with the use of `Isaac Automator <https://github.com/isaac-sim/IsaacAutomator>`__.
Isaac Automator allows for quick deployment of Isaac Sim and Isaac Lab onto the public clouds (AWS, GCP, Azure, and Alibaba Cloud are currently supported).

The result is a fully configured remote desktop cloud workstation, which can be used for development and testing of Isaac Lab within minutes and on a budget. Isaac Automator supports variety of GPU instances and stop-start functionality to save on cloud costs and a variety of tools to aid the workflow (like uploading and downloading data, autorun, deployment management, etc).


Installing Isaac Automator
--------------------------

For the most update-to-date and complete installation instructions, please refer to `Isaac Automator <https://github.com/isaac-sim/IsaacAutomator?tab=readme-ov-file#installation>`__.

To use Isaac Automator, first clone the repo:

.. code-block:: bash

   git clone https://github.com/isaac-sim/IsaacAutomator.git

Isaac Automator requires having ``docker`` pre-installed on the system.

* To install Docker, please follow the instructions for your operating system on the `Docker website`_.
* Follow the post-installation steps for Docker on the `post-installation steps`_ page. These steps allow you to run
  Docker without using ``sudo``.

Isaac Automator also requires obtaining a NGC API key.

* Get access to the `Isaac Sim container`_ by joining the NVIDIA Developer Program credentials.
* Generate your `NGC API key`_ to access locked container images from NVIDIA GPU Cloud (NGC).

  * This step requires you to create an NGC account if you do not already have one.
  * Once you have your generated API key, you need to log in to NGC
    from the terminal.

    .. code:: bash

         docker login nvcr.io

  * For the username, enter ``$oauthtoken`` exactly as shown. It is a special username that is used to
    authenticate with NGC.

    .. code:: text

        Username: $oauthtoken
        Password: <Your NGC API Key>


Running Isaac Automator
-----------------------

To run Isaac Automator, first build the Isaac Automator container:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         ./build

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         docker build --platform linux/x86_64 -t isa .

Next, enter the automator container:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         ./run

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         docker run --platform linux/x86_64 -it --rm -v .:/app isa bash

Next, run the deployed script for your preferred cloud:

.. code-block:: bash

   # AWS
   ./deploy-aws
   # Azure
   ./deploy-azure
   # GCP
   ./deploy-gcp
   # Alibaba Cloud
   ./deploy-alicloud

Follow the prompts for entering information regarding the environment setup and credentials.
Once successful, instructions for connecting to the cloud instance will be available in the terminal.
Connections can be made using SSH, noVCN, or NoMachine.

For details on the credentials and setup required for each cloud, please visit the
`Isaac Automator <https://github.com/isaac-sim/IsaacAutomator?tab=readme-ov-file#deploying-isaac-sim>`__
page for more instructions.


Running Isaac Lab on the Cloud
------------------------------

Once connected to the cloud instance, the desktop will have an icon showing ``isaaclab.sh``.
Launch the ``isaaclab.sh`` executable, which will open a new Terminal. Within the terminal,
Isaac Lab commands can be executed in the same way as running locally.

For example:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code-block:: bash

         ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Cartpole-v0

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code-block:: batch

         ./isaaclab.bat -p scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Cartpole-v0


Destroying a Development
-------------------------

To save costs, deployments can be destroyed when not being used.
This can be done from within the Automator container, which can be entered with command ``./run``.

To destroy a deployment, run:

.. code:: bash

   ./destroy <deployment-name>


.. _`Docker website`: https://docs.docker.com/desktop/install/linux-install/
.. _`post-installation steps`: https://docs.docker.com/engine/install/linux-postinstall/
.. _`Isaac Sim container`: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim
.. _`NGC API key`: https://docs.nvidia.com/ngc/gpu-cloud/ngc-user-guide/index.html#generating-api-key
