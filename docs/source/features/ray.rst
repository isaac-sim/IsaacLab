=================================
Ray Multi-Job Dispatch and Tuning
=================================

.. currentmodule:: omni.isaac.lab

Isaac Lab supports Ray for streamlining dispatching multiple training jobs (in parallel and in series),
and hyperparameter tuning, both on local and remote configurations.

.. attention::

  This functionality is experimental. It is recommended to read the entire guide prior to attempting to
  dispatch jobs or Tune runs.

Overview
--------

The Ray integration is useful for the following:

- Dispatching several training jobs in parallel or sequentially with minimal interaction
- Tuning hyperparameters; in parallel or sequentially with support for multiple GPUs and/or multiple GPU Nodes
- Using the same training setup everywhere (on cloud and local) with minimal overhead
- Resource Isolation for training jobs


**Installation**
----------------

The Ray functionality requires additional dependencies be installed.

To use Ray without Kubernetes, like on a local computer or VM,
``kubectl`` is not required. For use on Kubernetes clusters,
such as Google Kubernetes Engine or Amazon Elastic Kubernetes Service, ``kubectl`` is required, and can
be installed via the `Kubernetes website <https://kubernetes.io/docs/tasks/tools/>`_

The pythonic dependencies can be installed with:

.. code-block:: bash

  ./isaaclab.sh -p -m pip install ray[default, tune]==2.31.0
  ./isaaclab.sh -p -m pip install optuna bayesian-optimization

**Setup: Cluster Configuration**
--------------------------------

Select one of the following methods to create a Ray Cluster to accept and execute dispatched jobs.

Single-Node Ray Cluster (Recommended for Beginners)
'''''''''''''''''''''''''''''''''''''''''''''''''''
For use on a single machine (node) such as a local computer or VM, the
following command can be used start a ray server. This is compatible with
multiple-GPU machines. This Ray server will run indefinitely until it is stopped with ``CTRL + C``

.. code-block:: bash

  echo "import ray; ray.init(); import time; [time.sleep(10) for _ in iter(int, 1)]" | ./isaaclab.sh -p

KubeRay Clusters
''''''''''''''''
.. attention::
  The ``ray`` command should be modified to use Isaac python, which could be achieved in a fashion similar to
  ``sed -i "1i $(echo "#!/workspace/isaaclab/_isaac_sim/python.sh")" \
  /isaac-sim/kit/python/bin/ray && ln -s /isaac-sim/kit/python/bin/ray /usr/local/bin/ray``.

Google Cloud is currently the only platform tested, although
any cloud provider should work if one configures the following:

- An container registry (NGC, GCS artifact registry, AWS ECR, etc) with
  an Isaac Lab image configured to support Ray. See ``cluster_configs/Dockerfile`` to see how to modify the ``isaac-lab-base``
  container for Ray compatibility. Ray should use the isaac sim python shebang, and ``nvidia-smi``
  should work within the container. Be careful with the setup here as
  paths need to be configured correctly for everything to work. It's likely that
  the example dockerfile will work out of the box and can be pushed to the registry, as
  long as the base image has already been built as in the container guide
- A Kubernetes setup with available NVIDIA RTX (likely ``l4`` or ``l40``) GPU-passthrough node-pool resources,
  that has access to your container registry/storage bucket and has the Ray operator enabled with correct IAM
  permissions. This can be easily achieved with services such as Google GKE Autopilot or AWS EKS Fargate,
  provided that your account or organization has been granted a GPU-budget.
- A ``kuberay.yaml.ninja`` file that describes how to allocate resources (already included for
  Google Cloud)
- A storage bucket to dump experiment logs/checkpoints to, that the cluster has ``read/write`` access to.


Ray Clusters (Without Kubernetes)
'''''''''''''''''''''''''''''''''
.. attention::
  Modify the Ray command to use Isaac Python like in KubeRay Clusters, and follow the same
  steps for creating an image/cluster permissions/bucket access.

See the `Ray Clusters Overview <https://docs.ray.io/en/latest/cluster/getting-started.html>`_ or
`Anyscale <https://www.anyscale.com/product>`_ for more information

SLURM Ray Cluster
'''''''''''''''''
See the `Ray Community SLURM support <https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html#slurm-network-ray>`_
for more information. This functionality is theoretically possible, although the existing
Isaac SLURM support independent of Ray has been tested, unlike Ray SLURM.

**Dispatching Jobs and Tuning**
-------------------------------

Provided that there is a Ray cluster running with a correct configuration, select one of the following guides
that matches your Cluster configuration.

Single-Node Ray Cluster
'''''''''''''''''''''''
 Still being copied from README

Multiple-Node Ray Cluster
'''''''''''''''''''''''''
On a multiple-node Ray cluster, it is assumed that resources are homogeneous across workers (although not necessarily
the head Node).

  Still being copied from README

Multiple-Cluster Multiple-Node Ray
''''''''''''''''''''''''''''''''''
Multiple Clusters can be used to enable simultaneous training runs with hetereogeneous resource requirements

  Still being copied from README

**Cluster Cleanup**
'''''''''''''''''''

For the sake of conserving resources, and potentially freeing precious GPU resources for other people to use
on shared compute platforms, please destroy the Ray cluster after use. They can be easily
recreated! For KubeRay clusters, this can be done via

.. code-block:: bash

  kubectl get raycluster | egrep 'hyperparameter-tuner' | awk '{print $1}' | xargs kubectl delete raycluster
  kubectl delete secret bucket-access
