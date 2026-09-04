.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _docker-cloud:

Docker/Cloud
============

Use this page after completing the minimal container or cloud setup in :ref:`isaaclab-installation-root`.
It contains the operational details for developing in Docker, retaining artifacts, extending images,
submitting jobs to a cluster, and managing remote workstations.

Isaac Lab provides a Dockerfile and Compose configuration that layer Isaac Lab onto the NVIDIA Isaac
Sim container. The resulting image is reproducible across Docker hosts and can be converted to an
Apptainer/Singularity image for shared HPC systems. For public-cloud workstations, Isaac Automator
provisions Isaac Sim and Isaac Lab on AWS, GCP, Azure, and Alibaba Cloud.

Docker guide
------------

.. include:: include/docker_details.inc

Worked Docker example
---------------------

.. include:: include/docker_example_details.inc

Clusters
--------

.. include:: include/cluster_details.inc

.. _docker-cloud-cloud:

Cloud workstations
------------------

.. include:: include/cloud_details.inc
