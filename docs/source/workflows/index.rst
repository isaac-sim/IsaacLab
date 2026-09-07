.. _workflows:

Workflows
=========

A workflow is a packaged way to run Isaac Lab somewhere other than a local Python environment. The
task code, the configuration, and the command line stay the same; only the environment that executes
them changes. Use this section once Isaac Lab runs locally through
:ref:`isaaclab-installation-root` and you need a reproducible environment, a job scheduler, or a
remote GPU.

Every workflow below starts from the same container image, so they build on each other: the cluster
workflow converts that image to Apptainer, and the cloud workflow provisions a machine that runs it.

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card:: **Docker**
      :link: deployment-docker
      :link-type: ref

      Build and run Isaac Lab in a container for a reproducible environment on any Docker host.
      **Start here** -- the other two workflows build on this image.

   .. grid-item-card:: **HPC clusters**
      :link: deployment-cluster
      :link-type: ref

      Convert the container to an Apptainer image and submit jobs through SLURM, PBS, or OSMO.

   .. grid-item-card:: **Cloud workstations**
      :link: docker-cloud-cloud
      :link-type: ref

      Provision a GPU workstation on AWS, GCP, Azure, or Alibaba Cloud with Isaac Automator.

.. toctree::
   :maxdepth: 2

   docker/index
