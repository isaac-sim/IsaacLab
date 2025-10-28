.. _container-deployment:

Container Deployment
====================

Docker is a tool that allows for the creation of containers, which are isolated environments that can
be used to run applications. They are useful for ensuring that an application can run on any machine
that has Docker installed, regardless of the host machine's operating system or installed libraries.

We include a Dockerfile and docker-compose.yaml file that can be used to build a Docker image that
contains Isaac Lab and all of its dependencies. This image can then be used to run Isaac Lab in a container.
The Dockerfile is based on the Isaac Sim image provided by NVIDIA, which includes the Omniverse
application launcher and the Isaac Sim application. The Dockerfile installs Isaac Lab and its dependencies
on top of this image.

Cloning the Repository
----------------------

Before building the container, clone the Isaac Lab repository (if not already done):

.. tab-set::

   .. tab-item:: SSH

      .. code:: bash

         git clone git@github.com:isaac-sim/IsaacLab.git

   .. tab-item:: HTTPS

      .. code:: bash

         git clone https://github.com/isaac-sim/IsaacLab.git

Next Steps
----------

After cloning, you can choose the deployment workflow that fits your needs:

- :doc:`docker`

  - Learn how to build, configure, and run Isaac Lab in Docker containers.
  - Explains the repository's ``docker/`` setup, the ``container.py`` helper script, mounted volumes,
    image extensions (like ROS 2), and optional CloudXR streaming support.
  - Covers running pre-built Isaac Lab containers from NVIDIA NGC for headless training.

- :doc:`run_docker_example`

  - Learn how to run a development workflow inside the Isaac Lab Docker container.
  - Demonstrates building the container, entering it, executing a sample Python script (`log_time.py`),
    and retrieving logs using mounted volumes.
  - Highlights bind-mounted directories for live code editing and explains how to stop or remove the container
    while keeping the image and artifacts.

- :doc:`cluster`

  - Learn how to run Isaac Lab on high-performance computing (HPC) clusters.
  - Explains how to export the Docker image to a Singularity (Apptainer) image, configure cluster-specific parameters,
    and submit jobs using common workload managers (SLURM or PBS).
  - Includes tested workflows for ETH Zurich's Euler cluster and IIT Genoa's Franklin cluster,
    with notes on adapting to other environments.

- :doc:`cloudxr_teleoperation_cluster`

  - Deploy CloudXR Teleoperation for Isaac Lab on a Kubernetes cluster.
  - Covers system requirements, software dependencies, and preparation steps including RBAC permissions.
  - Demonstrates how to install and verify the Helm chart, run the pod, and uninstall it.


.. toctree::
   :maxdepth: 1
   :hidden:

   docker
   run_docker_example
   cluster
   cloudxr_teleoperation_cluster
