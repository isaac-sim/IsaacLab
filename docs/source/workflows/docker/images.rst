.. _docker-images:

Container images
================

Each :ref:`profile <deployment-docker>` builds a different image. They differ in what simulation and
rendering stack they carry, which decides how large they are and what they can run. This page
describes the images themselves; to build or run one, name its profile in the
:ref:`container lifecycle <container-lifecycle>`.

Choosing an image
-----------------

.. list-table::
   :header-rows: 1
   :widths: 12 40 48

   * - Profile
     - Contains
     - Use it when
   * - ``base``
     - Isaac Lab on the NVIDIA Isaac Sim image, with Kit and RTX rendering
     - You want the default environment, GUI rendering, or anything that needs Isaac Sim
   * - ``ros2``
     - The ``base`` image plus ROS 2 Humble
     - You are bridging Isaac Lab to ROS 2 nodes
   * - ``kitless``
     - Ubuntu 24.04, Python 3.12, Newton and OVPhysX physics, OVRTX rendering, four RL libraries
     - You are training with Newton and want the smallest image; also the only one with no Isaac Sim EULA

Base image
----------

``Dockerfile.base`` overlays Isaac Lab's dependencies onto the Isaac Sim container, so it inherits
Kit, RTX rendering, and the Isaac Sim asset pipeline. The Isaac Sim version it builds against is set
by ``ISAACSIM_VERSION`` in ``.env.base``; the other variables in that file control paths inside the
container.

ROS 2 image
-----------

``Dockerfile.ros2`` installs ROS 2 Humble from an `apt package`_ and sources it in the runtime
user's ``.bashrc``. ``ROS_APT_PACKAGE`` in ``.env.ros2`` selects the exact version, defaulting to
``ros-base``. The image defaults to the ``FastRTPS`` middleware; ``CycloneDDS`` is also supported, and
both can be `tuned`_ through their ``.xml`` files under ``docker/.ros``. See `various middleware`_
for the trade-offs.

.. dropdown:: Parameters in .env.ros2
   :icon: code

   .. literalinclude:: ../../../../docker/.env.ros2
      :language: bash

Kit-less image
--------------

The kit-less image drops Isaac Sim entirely and builds on Ubuntu 24.04 with Python 3.12. It carries
Newton physics, OVPhysX physics, OVRTX rendering, and the four core RL frameworks: RL Games, RSL-RL,
Stable-Baselines3, and SKRL. Every Newton viewer -- ``newton``, ``viser``, and ``rerun`` -- is
included, so any of them can be requested with ``--viz``. No visualizer is selected by default, so
training runs headless unless you ask for one. The ``kit`` visualizer is the exception: it comes from
Omniverse Kit, which this image does not contain.

Unlike the other two, it bind-mounts ``apps`` in addition to ``source``, ``scripts``, ``tools``, and
``docs``, and it keeps a uv cache and a Warp cache of its own.

You do not have to build it locally. It is published alongside each Isaac Lab release in the same
registry repository, distinguished by a ``-kitless`` tag suffix:

.. code:: bash

    docker pull nvcr.io/nvidia/isaac-lab:<version>-kitless

Running it directly, outside the Compose workflow, is a convenient way to verify the image and your
NVIDIA Container Toolkit setup. ``--interactive`` and ``--tty`` keep the container's shell alive so
that several commands can share it:

.. code:: bash

    docker run --name isaac-lab-kitless --detach --interactive --tty --gpus all --network host \
        nvcr.io/nvidia/isaac-lab:<version>-kitless

    docker exec isaac-lab-kitless \
        isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct \
        --num_envs 16 presets=newton_mjwarp --max_iterations 5

Swap ``train`` for ``play --checkpoint latest --viz viser`` to replay the result; the Viser web
visualizer is reachable through the host network. To build the image from the checkout instead of
pulling it, run ``docker build --file docker/Dockerfile.kitless --tag isaac-lab-kitless .``.

Only ``linux/amd64`` images are published. The Dockerfile itself is architecture-agnostic, but the
published manifest is limited to the architecture that is validated in CI.

Pre-built image from NGC
------------------------

A minimal pre-built container carries a small set of Isaac Sim and Omniverse dependencies with Isaac
Lab already built in, under ``/workspace/IsaacLab``. It is **headless only** -- it does not support
X11 forwarding or a GUI -- so use it for headless training and build your own image for anything
else.

.. note::

  Currently, we only provide docker images with every major release of Isaac Lab.
  For example, we provide the docker image for release 2.0.0 and 2.1.0, but not 2.0.2.
  In the future, we will provide docker images for every minor release of Isaac Lab.

Because it is run outside Compose, the Isaac Sim cache directories have to be mounted by hand,
otherwise every start recompiles shaders:

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
     nvcr.io/nvidia/isaac-lab:3.0.0-beta2

To render through X11, run ``xhost +`` on the host and add ``-e DISPLAY`` and
``-v $HOME/.Xauthority:/root/.Xauthority`` to the command above.

.. attention::

  Images from 3.0.0-beta2 onward run as a **non-root** user, so those bind-mount directories must be
  writable by uid/gid 1000. Docker creates any missing one as ``root``, which the runtime user cannot
  write to, producing errors such as
  ``PermissionError: [Errno 13] Permission denied: '/root/.local/share/ov/data/exts'``.
  Create them first:

  .. code:: bash

     mkdir -p ~/docker/isaac-sim/{cache/kit,cache/ov,cache/pip,cache/glcache,cache/computecache,logs,data,documents}
     sudo chown -R 1000:1000 ~/docker/isaac-sim

Runtime user
------------

The base, ROS 2, cuRobo, and kit-less images all run as a non-root user with uid/gid 1000, which
keeps bind-mounted workspaces writable on GitHub runners. When running one of these images directly
with ``docker run`` from a host account whose uid differs, pass ``--user "$(id -u):1000"`` so that
new files on bind mounts belong to your host user while the runtime home stays accessible.

Python interpreter
------------------

Every image installs from ``uv.lock`` into a Python 3.12 virtual environment at
``/opt/isaaclab-venv``. On the Isaac Sim-based images the environment is built on Isaac Sim's own
interpreter, and Isaac Sim itself stays outside it, reached through the ``_isaac_sim`` symlink;
``isaaclab.sh`` puts it on the path. In either case ``python`` on the container's ``PATH`` resolves
to the right one, so scripts are run the same way regardless of image.

.. _`apt package`: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html#install-ros-2-packages
.. _`various middleware`: https://docs.ros.org/en/humble/How-To-Guides/Working-with-multiple-RMW-implementations.html
.. _`tuned`: https://docs.ros.org/en/foxy/How-To-Guides/DDS-tuning.html
