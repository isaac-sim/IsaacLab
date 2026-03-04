.. _deployment-docker:


Docker Guide
============

.. caution::

    Due to the dependency on Isaac Sim docker image, by running this container you are implicitly
    agreeing to the `NVIDIA Software License Agreement`_. If you do not agree to the EULA, do not run this container.

Setup Instructions
------------------

.. note::

    The following steps are taken from the Isaac Sim documentation on `container installation`_.
    They have been added here for the sake of completeness.


Docker and Docker Compose
~~~~~~~~~~~~~~~~~~~~~~~~~

We have tested the container using Docker Engine version 26.0.0 and Docker Compose version 2.25.0
We recommend using these versions or newer.

* To install Docker, please follow the instructions for your operating system on the `Docker website`_.
* To install Docker Compose, please follow the instructions for your operating system on the `docker compose`_ page.
* Follow the post-installation steps for Docker on the `post-installation steps`_ page. These steps allow you to run
  Docker without using ``sudo``.
* To build and run GPU-accelerated containers, you also need install the `NVIDIA Container Toolkit`_.
  Please follow the instructions on the `Container Toolkit website`_ for installation steps.

.. note::

    Due to limitations with `snap <https://snapcraft.io/docs/home-outside-home>`_, please make sure
    the Isaac Lab directory is placed under the ``/home`` directory tree when using docker.


Directory Organization
----------------------

The root of the Isaac Lab repository contains the ``docker`` directory that has various files and scripts
needed to run Isaac Lab inside a Docker container. A subset of these are summarized below:

* **Dockerfile.base**: Defines the base Isaac Lab image by overlaying its dependencies onto the Isaac Sim Docker image.
  Dockerfiles which end with something else, (i.e. ``Dockerfile.ros2``) build an `image extension <#isaac-lab-image-extensions>`_.
* **docker-compose.yaml**: Creates mounts to allow direct editing of Isaac Lab code from the host machine that runs
  the container. It also creates several named volumes such as ``isaac-cache-kit`` to
  store frequently reused resources compiled by Isaac Sim, such as shaders, and to retain logs, data, and documents.
* **.env.base**: Stores environment variables required for the ``base`` build process and the container itself. ``.env``
  files which end with something else (i.e. ``.env.ros2``) define these for `image extension <#isaac-lab-image-extensions>`_.
* **docker-compose.cloudxr-runtime.patch.yaml**: A patch file that is applied to enable CloudXR Runtime support for
  streaming to compatible XR devices. It defines services and volumes for CloudXR Runtime and the base.
* **.env.cloudxr-runtime**: Environment variables for the CloudXR Runtime support.
* **container.py**: A utility script that interfaces with tools in ``utils`` to configure and build the image,
  and run and interact with the container.

Running the Container
---------------------

.. note::

    The docker container copies all the files from the repository into the container at the
    location ``/workspace/isaaclab`` at build time. This means that any changes made to the files in the container would not
    normally be reflected in the repository after the image has been built, i.e. after ``./container.py start`` is run.

    For a faster development cycle, we mount the following directories in the Isaac Lab repository into the container
    so that you can edit their files from the host machine:

    * **IsaacLab/source**: This is the directory that contains the Isaac Lab source code.
    * **IsaacLab/docs**: This is the directory that contains the source code for Isaac Lab documentation. This is overlaid except
      for the ``_build`` subdirectory where build artifacts are stored.


The script ``container.py`` parallels basic ``docker compose`` commands. Each can accept an `image extension argument <#isaac-lab-image-extensions>`_,
or else they will default to the ``base`` image extension. These commands are:

* **build**: This builds the image for the given profile. It does not bring up the container.
* **start**: This builds the image and brings up the container in detached mode (i.e. in the background).
* **enter**: This begins a new bash process in an existing Isaac Lab container, and which can be exited
  without bringing down the container.
* **config**: This outputs the compose.yaml which would be result from the inputs given to ``container.py start``. This command is useful
  for debugging a compose configuration.
* **copy**: This copies the ``logs``, ``data_storage`` and ``docs/_build`` artifacts, from the ``isaac-lab-logs``, ``isaac-lab-data`` and ``isaac-lab-docs``
  volumes respectively, to the ``docker/artifacts`` directory. These artifacts persist between docker container instances and are shared between image extensions.
* **stop**: This brings down the container and removes it.

The following shows how to launch the container in a detached state and enter it:

.. code:: bash

    # Launch the container in detached mode
    # We don't pass an image extension arg, so it defaults to 'base'
    ./docker/container.py start

    # If we want to add .env or .yaml files to customize our compose config,
    # we can simply specify them in the same manner as the compose cli
    # ./docker/container.py start --file my-compose.yaml --env-file .env.my-vars

    # Enter the container
    # We pass 'base' explicitly, but if we hadn't it would default to 'base'
    ./docker/container.py enter base

To copy files from the base container to the host machine, you can use the following command:

.. code:: bash

    # Copy the file /workspace/isaaclab/logs to the current directory
    docker cp isaac-lab-base:/workspace/isaaclab/logs .

The script ``container.py`` provides a wrapper around this command to copy the ``logs`` , ``data_storage`` and ``docs/_build``
directories to the ``docker/artifacts`` directory. This is useful for copying the logs, data and documentation:

.. code:: bash

    # stop the container
    ./docker/container.py stop


CloudXR Runtime Support
~~~~~~~~~~~~~~~~~~~~~~~

To enable CloudXR Runtime for streaming to compatible XR devices, you need to apply the patch file
``docker-compose.cloudxr-runtime.patch.yaml`` to run CloudXR Runtime container. The patch file defines services and
volumes for CloudXR Runtime and base. The environment variables required for CloudXR Runtime are specified in the
``.env.cloudxr-runtime`` file. To start or stop the CloudXR runtime container with base, use the following command:

.. code:: bash

    # Start CloudXR Runtime container with base.
    ./docker/container.py start --files docker-compose.cloudxr-runtime.patch.yaml --env-file .env.cloudxr-runtime

    # Stop CloudXR Runtime container and base.
    ./docker/container.py stop --files docker-compose.cloudxr-runtime.patch.yaml --env-file .env.cloudxr-runtime


X11 forwarding
~~~~~~~~~~~~~~

The container supports X11 forwarding, which allows the user to run GUI applications from the container
and display them on the host machine.

The first time a container is started with ``./docker/container.py start``, the script prompts
the user whether to activate X11 forwarding. This will create a file at ``docker/.container.cfg``
to store the user's choice for future runs.

If you want to change the choice, you can set the parameter ``X11_FORWARDING_ENABLED`` to '0' or '1'
in the ``docker/.container.cfg`` file to disable or enable X11 forwarding, respectively. After that, you need to
re-build the container by running ``./docker/container.py start``. The rebuilding process ensures that the changes
are applied to the container. Otherwise, the changes will not take effect.

After the container is started, you can enter the container and run GUI applications from it with X11 forwarding enabled.
The display will be forwarded to the host machine.


Python Interpreter
~~~~~~~~~~~~~~~~~~

The container uses the Python interpreter provided by Isaac Sim. This interpreter is located at
``/isaac-sim/python.sh``. We set aliases inside the container to make it easier to run the Python
interpreter. You can use the following commands to run the Python interpreter:

.. code:: bash

    # Run the Python interpreter -> points to /isaac-sim/python.sh
    python


Understanding the mounted volumes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``docker-compose.yaml`` file creates several named volumes that are mounted to the container.
These are summarized below:

.. list-table::
   :header-rows: 1
   :widths: 23 45 32

   * - Volume Name
     - Description
     - Container Path
   * - isaac-cache-kit
     - Stores cached Kit resources
     - /isaac-sim/kit/cache
   * - isaac-cache-ov
     - Stores cached OV resources
     - /root/.cache/ov
   * - isaac-cache-pip
     - Stores cached pip resources
     - /root/.cache/pip
   * - isaac-cache-gl
     - Stores cached GLCache resources
     - /root/.cache/nvidia/GLCache
   * - isaac-cache-compute
     - Stores cached compute resources
     - /root/.nv/ComputeCache
   * - isaac-logs
     - Stores logs generated by Omniverse
     - /root/.nvidia-omniverse/logs
   * - isaac-carb-logs
     - Stores logs generated by carb
     - /isaac-sim/kit/logs/Kit/Isaac-Sim
   * - isaac-data
     - Stores data generated by Omniverse
     - /root/.local/share/ov/data
   * - isaac-docs
     - Stores documents generated by Omniverse
     - /root/Documents
   * - isaac-lab-docs
     - Stores documentation of Isaac Lab when built inside the container
     - /workspace/isaaclab/docs/_build
   * - isaac-lab-logs
     - Stores logs generated by Isaac Lab workflows when run inside the container
     - /workspace/isaaclab/logs
   * - isaac-lab-data
     - Stores whatever data users may want to preserve between container runs
     - /workspace/isaaclab/data_storage

To view the contents of these volumes, you can use the following command:

.. code:: bash

    # list all volumes
    docker volume ls
    # inspect a specific volume, e.g. isaac-cache-kit
    docker volume inspect isaac-cache-kit



Isaac Lab Image Extensions
--------------------------

The produced image depends on the arguments passed to ``container.py start`` and ``container.py stop``. These
commands accept an image extension parameter as an additional argument. If no argument is passed, then this
parameter defaults to ``base``. Currently, the only valid values are (``base``, ``ros2``).
Only one image extension can be passed at a time.  The produced image and container will be named
``isaac-lab-${profile}``, where ``${profile}`` is the image extension name.

``suffix`` is an optional string argument to ``container.py`` that specifies a docker image and
container name suffix, which can be useful for development purposes. By default ``${suffix}`` is the empty string.
If ``${suffix}`` is a nonempty string, then the produced docker image and container will be named
``isaac-lab-${profile}-${suffix}``, where a hyphen is inserted between ``${profile}`` and ``${suffix}`` in
the name. ``suffix`` should not be used with cluster deployments.

.. code:: bash

    # start base by default, named isaac-lab-base
    ./docker/container.py start
    # stop base explicitly, named isaac-lab-base
    ./docker/container.py stop base
    # start ros2 container named isaac-lab-ros2
    ./docker/container.py start ros2
    # stop ros2 container named isaac-lab-ros2
    ./docker/container.py stop ros2

    # start base container named isaac-lab-base-custom
    ./docker/container.py start base --suffix custom
    # stop base container named isaac-lab-base-custom
    ./docker/container.py stop base --suffix custom
    # start ros2 container named isaac-lab-ros2-custom
    ./docker/container.py start ros2 --suffix custom
    # stop ros2 container named isaac-lab-ros2-custom
    ./docker/container.py stop ros2 --suffix custom

The passed image extension argument will build the image defined in ``Dockerfile.${image_extension}``,
with the corresponding `profile`_ in the ``docker-compose.yaml`` and the envars from ``.env.${image_extension}``
in addition to the ``.env.base``, if any.

ROS2 Image Extension
~~~~~~~~~~~~~~~~~~~~

In ``Dockerfile.ros2``, the container installs ROS2 Humble via an `apt package`_, and it is sourced in the ``.bashrc``.
The exact version is specified by the variable ``ROS_APT_PACKAGE`` in the ``.env.ros2`` file,
defaulting to ``ros-base``. Other relevant ROS2 variables are also specified in the ``.env.ros2`` file,
including variables defining the `various middleware`_ options.

The container defaults to ``FastRTPS``, but ``CylconeDDS`` is also supported. Each of these middlewares can be
`tuned`_ using their corresponding ``.xml`` files under ``docker/.ros``.


.. dropdown:: Parameters for ROS2 Image Extension
   :icon: code

   .. literalinclude:: ../../../docker/.env.ros2
      :language: bash


Running Pre-Built Isaac Lab Container
-------------------------------------

In Isaac Lab 2.0 release, we introduced a minimal pre-built container that contains a very minimal set
of Isaac Sim and Omniverse dependencies, along with Isaac Lab 2.0 pre-built into the container.
This container allows users to pull the container directly from NGC without requiring a local build of
the docker image. The Isaac Lab source code will be available in this container under ``/workspace/IsaacLab``.

This container is designed for running **headless** only and does not allow for X11 forwarding or running
with the GUI. Please only use this container for headless training. For other use cases, we recommend
following the above steps to build your own Isaac Lab docker image.

.. note::

  Currently, we only provide docker images with every major release of Isaac Lab.
  For example, we provide the docker image for release 2.0.0 and 2.1.0, but not 2.0.2.
  In the future, we will provide docker images for every minor release of Isaac Lab.

To pull the minimal Isaac Lab container, run:

.. code:: bash

  docker pull nvcr.io/nvidia/isaac-lab:3.0.0

To run the Isaac Lab container with an interactive bash session, run:

.. code:: bash

  docker run --name isaac-lab --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
     -e "PRIVACY_CONSENT=Y" \
     -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
     -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
     -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
     -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
     -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
     -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
     -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
     -v ~/docker/isaac-sim/documents:/root/Documents:rw \
     nvcr.io/nvidia/isaac-lab:3.0.0

To enable rendering through X11 forwarding, run:

.. code:: bash

  xhost +
  docker run --name isaac-lab --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
     -e "PRIVACY_CONSENT=Y" \
     -e DISPLAY \
     -v $HOME/.Xauthority:/root/.Xauthority \
     -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
     -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
     -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
     -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
     -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
     -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
     -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
     -v ~/docker/isaac-sim/documents:/root/Documents:rw \
     nvcr.io/nvidia/isaac-lab:3.0.0

To run an example within the container, run:

.. code:: bash

  ./isaaclab.sh -p scripts/tutorials/00_sim/log_time.py --headless


.. _`NVIDIA Software License Agreement`: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement
.. _`container installation`: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html
.. _`Docker website`: https://docs.docker.com/desktop/install/linux-install/
.. _`docker compose`: https://docs.docker.com/compose/install/linux/#install-using-the-repository
.. _`NVIDIA Container Toolkit`: https://github.com/NVIDIA/nvidia-container-toolkit
.. _`Container Toolkit website`: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
.. _`post-installation steps`: https://docs.docker.com/engine/install/linux-postinstall/
.. _`Isaac Sim container`: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim
.. _`NGC API key`: https://docs.nvidia.com/ngc/gpu-cloud/ngc-user-guide/index.html#generating-api-key
.. _`several streaming clients`: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html
.. _`known issue`: https://forums.developer.nvidia.com/t/unable-to-use-webrtc-when-i-run-runheadless-webrtc-sh-in-remote-headless-container/222916
.. _`profile`: https://docs.docker.com/compose/compose-file/15-profiles/
.. _`apt package`: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html#install-ros-2-packages
.. _`various middleware`: https://docs.ros.org/en/humble/How-To-Guides/Working-with-multiple-RMW-implementations.html
.. _`tuned`: https://docs.ros.org/en/foxy/How-To-Guides/DDS-tuning.html
