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

The following guides provide instructions for building the Docker image and running Isaac Lab in a
container.

.. toctree::
  :maxdepth: 1

  docker
  cluster
  cloudxr_teleoperation_cluster
  run_docker_example
