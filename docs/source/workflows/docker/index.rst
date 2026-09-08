.. _docker-cloud:
.. _deployment-docker:

The container.py workflow
=========================

``docker/container.py`` is the single entry point for running Isaac Lab in a container. It wraps
``docker compose`` so that you do not have to remember which Dockerfile, environment files, and
volume mounts belong together -- you name a *profile*, and the script assembles the rest.

The image it builds is also the basis for the other workflows: :ref:`deployment-cluster` converts it
to an Apptainer image, and :ref:`docker-cloud-cloud` provisions a machine that runs it.

.. caution::

    The standard Isaac Lab container depends on the Isaac Sim Docker image. By running that container, you are
    implicitly agreeing to the `NVIDIA Software License Agreement`_. If you do not agree to the EULA, do not run
    that container. The ``kitless`` image contains neither Isaac Sim nor Kit.

Prerequisites
-------------

Install `Docker Engine <https://docs.docker.com/engine/install/>`__, `Docker Compose
<https://docs.docker.com/compose/install/>`__, and the `NVIDIA Container Toolkit
<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`__.
The container is tested with Docker Engine 26.0.0 and Docker Compose 2.25.0; use these versions or
newer. Follow the `post-installation steps`_ so that Docker runs without ``sudo``.

The Isaac Sim documentation on `container installation`_ covers the same prerequisites in more
detail, including how to obtain access to the Isaac Sim image.

.. note::

    Due to limitations with `snap <https://snapcraft.io/docs/home-outside-home>`_, please make sure
    the Isaac Lab directory is placed under the ``/home`` directory tree when using docker.

Container profiles
------------------

A profile is a `Docker Compose profile <https://docs.docker.com/compose/how-tos/profiles/>`__:
``docker-compose.yaml`` tags each service with one, and Compose starts only the services whose
profile is active. Naming a profile therefore selects a matched set -- one service, one Dockerfile,
one set of environment files, and one image name:

.. list-table::
   :header-rows: 1
   :widths: 12 20 20 24 24

   * - Profile
     - Compose service
     - Dockerfile
     - Environment files
     - Built on top of
   * - ``base``
     - ``isaac-lab-base``
     - ``Dockerfile.base``
     - ``.env.base``
     - the Isaac Sim image
   * - ``ros2``
     - ``isaac-lab-ros2``
     - ``Dockerfile.ros2``
     - ``.env.base`` + ``.env.ros2``
     - the ``base`` image, built first
   * - ``kitless``
     - ``isaac-lab-kitless``
     - ``Dockerfile.kitless``
     - ``.env.kitless``
     - nothing -- standalone, no Isaac Sim

Every command below takes the profile as its first positional argument and defaults to ``base``.
Only one profile applies at a time, and the resulting image and container are both named
``isaac-lab-<profile>``. Pass ``--suffix`` to append a name suffix when you want several variants of
the same profile side by side.

For what each image actually contains and how to choose between them, see :ref:`docker-images`.

.. _container-lifecycle:

The container lifecycle
-----------------------

This is the whole workflow. Append a profile to any command to target something other than ``base``:

.. code:: bash

    ./docker/container.py start     # build the image, then start the container detached
    ./docker/container.py enter     # open a shell inside the running container
    #   ... work here; see the note below on what is editable from the host ...
    ./docker/container.py copy      # copy logs, data, and built docs out to docker/artifacts/
    ./docker/container.py stop      # stop the container and remove it

.. note::

    The image copies the repository to ``/workspace/isaaclab`` at build time, so edits made after the
    build are not picked up automatically. To keep the development loop fast, the compose file
    bind-mounts ``source``, ``scripts``, ``docs``, and ``tools`` from the host, so changes to those
    directories appear inside the container immediately. Everything else requires a rebuild.

``container.py`` command reference
----------------------------------

Generated from ``docker/container.py``, so it always matches the installed script. The same
information is available from ``./docker/container.py --help``.

.. isaaclab-container-cli::
   :section: commands

Every command accepts the following arguments:

.. isaaclab-container-cli::
   :section: options

Extending the Compose configuration
-----------------------------------

``--files`` and ``--env-files`` merge extra Compose and environment files into the generated
configuration, which is how optional components are layered on without editing the checked-in files.
Streaming to XR devices is the worked example -- it adds the CloudXR Runtime service alongside
``base``:

.. code:: bash

    ./docker/container.py start --files docker-compose.cloudxr-runtime.patch.yaml --env-file .env.cloudxr-runtime

Stop it with the same arguments. The teleoperation setup, firewall rules, and client connection
steps are covered in :ref:`cloudxr-teleoperation`. Use ``./docker/container.py config`` to print the
merged result when a combination does not behave as expected.

What persists between runs
--------------------------

The compose file declares named volumes so that Isaac Sim caches, logs, and your own data survive
``stop``. The caches are why the second start is much faster than the first.

``container.py copy`` extracts the three volumes you are most likely to want on the host --
``logs``, ``data_storage``, and ``docs/_build`` -- into ``docker/artifacts``. For anything else, use
``docker cp``, for example ``docker cp isaac-lab-base:/workspace/isaaclab/logs .``.

If you are upgrading from an Isaac Lab image that ran as ``root``, the existing volumes still hold
root-owned files that the current uid/gid 1000 runtime user cannot write. Copy out anything worth
keeping, then recreate them:

.. code:: bash

    docker compose --file docker-compose.yaml --profile base --env-file .env.base down --volumes

.. dropdown:: All named volumes and their container paths
   :icon: file-directory

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
      * - isaac-cache-uv
        - Stores uv downloads for the kit-less profile
        - /home/isaaclab/.cache/uv
      * - isaac-cache-warp
        - Stores Warp kernels for the kit-less profile
        - /home/isaaclab/.cache/warp
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

   Inspect one from the host with ``docker volume inspect isaac-cache-kit``. The ``kitless`` profile
   uses the uv and Warp caches and shares the documentation, logs, and data volumes with the others.

Display forwarding with X11
---------------------------

X11 forwarding lets GUI applications started inside the container display on the host. The first
``start`` asks whether to enable it and records the answer in ``docker/.container.cfg``. To change it
later, set ``X11_FORWARDING_ENABLED`` to ``0`` or ``1`` in that file and run ``start`` again -- the
rebuild is what applies the change.

.. toctree::
   :maxdepth: 1

   images
   example
   cluster
   cloud

.. _`NVIDIA Software License Agreement`: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement
.. _`container installation`: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html
.. _`post-installation steps`: https://docs.docker.com/engine/install/linux-postinstall/
