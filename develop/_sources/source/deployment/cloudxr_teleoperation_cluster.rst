.. _cloudxr-teleoperation-cluster:

Deploying CloudXR Teleoperation on Kubernetes
=============================================

.. currentmodule:: isaaclab

This section explains how to deploy CloudXR Teleoperation for Isaac Lab on a Kubernetes (K8s) cluster.

.. _k8s-system-requirements:

System Requirements
-------------------

* **Minimum requirement**: Kubernetes cluster with a node that has at least 1 NVIDIA RTX PRO 6000 / L40 GPU or equivalent
* **Recommended requirement**: Kubernetes cluster with a node that has at least 2 RTX PRO 6000 / L40 GPUs or equivalent

.. note::
   If you are using DGX Spark, check :ref:`dgx-spark-limitations` for compatibility.

Software Dependencies
---------------------

* ``kubectl`` on your host computer

  * If you use MicroK8s, you already have ``microk8s kubectl``
  * Otherwise follow the `official kubectl installation guide <https://kubernetes.io/docs/tasks/tools/#kubectl>`_

* ``helm`` on your host computer

  * If you use MicroK8s, you already have ``microk8s helm``
  * Otherwise follow the `official Helm installation guide <https://helm.sh/docs/intro/install/>`_

* Access to NGC public registry from your Kubernetes cluster, in particular these container images:

  * ``https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-lab``
  * ``https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cloudxr-runtime``

* NVIDIA GPU Operator or equivalent installed in your Kubernetes cluster to expose NVIDIA GPUs
* NVIDIA Container Toolkit installed on the nodes of your Kubernetes cluster

Preparation
-----------

On your host computer, you should have already configured ``kubectl`` to access your Kubernetes cluster. To validate, run the following command and verify it returns your nodes correctly:

.. code:: bash

   kubectl get node

If you are installing this to your own Kubernetes cluster instead of using the setup described in the :ref:`k8s-appendix`, your role in the K8s cluster should have at least the following RBAC permissions:

.. code:: yaml

   rules:
   - apiGroups: [""]
     resources: ["configmaps"]
     verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
   - apiGroups: ["apps"]
     resources: ["deployments", "replicasets"]
     verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
   - apiGroups: [""]
     resources: ["pods"]
     verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
   - apiGroups: [""]
     resources: ["services"]
     verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]

.. _k8s-installation:

Installation
------------

.. note::

   The following steps are verified on a MicroK8s cluster with GPU Operator installed (see configurations in the :ref:`k8s-appendix`). You can configure your own K8s cluster accordingly if you encounter issues.

#. Download the Helm chart from NGC (get your NGC API key based on the `public guide <https://docs.nvidia.com/ngc/ngc-overview/index.html#generating-api-key>`_):

   .. code:: bash

      helm fetch https://helm.ngc.nvidia.com/nvidia/charts/isaac-lab-teleop-2.3.0.tgz \
        --username='$oauthtoken' \
        --password=<your-ngc-api-key>

#. Install and run the CloudXR Teleoperation for Isaac Lab pod in the default namespace, consuming all host GPUs:

   .. code:: bash

      helm upgrade --install hello-isaac-teleop isaac-lab-teleop-2.3.0.tgz \
        --set fullnameOverride=hello-isaac-teleop \
        --set hostNetwork="true"

   .. note::

      You can remove the need for host network by creating an external LoadBalancer VIP (e.g., with MetalLB), and setting the environment variable ``NV_CXR_ENDPOINT_IP`` when deploying the Helm chart:

      .. code:: yaml

         # local_values.yml file example:
         fullnameOverride: hello-isaac-teleop
         streamer:
           extraEnvs:
             - name: NV_CXR_ENDPOINT_IP
               value: "<your external LoadBalancer VIP>"
             - name: ACCEPT_EULA
               value: "Y"

      .. code:: bash

         # command
         helm upgrade --install --values local_values.yml \
           hello-isaac-teleop isaac-lab-teleop-2.3.0.tgz

#. Verify the deployment is completed:

   .. code:: bash

      kubectl wait --for=condition=available --timeout=300s \
        deployment/hello-isaac-teleop

   After the pod is running, it might take approximately 5-8 minutes to complete loading assets and start streaming.

Uninstallation
--------------

You can uninstall by simply running:

.. code:: bash

   helm uninstall hello-isaac-teleop

.. _k8s-appendix:

Appendix: Setting Up a Local K8s Cluster with MicroK8s
------------------------------------------------------

Your local workstation should have the NVIDIA Container Toolkit and its dependencies installed. Otherwise, the following setup will not work.

Cleaning Up Existing Installations (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # Clean up the system to ensure we start fresh
   sudo snap remove microk8s
   sudo snap remove helm
   sudo apt-get remove docker-ce docker-ce-cli containerd.io
   # If you have snap docker installed, remove it as well
   sudo snap remove docker

Installing MicroK8s
~~~~~~~~~~~~~~~~~~~

.. code:: bash

   sudo snap install microk8s --classic

Installing NVIDIA GPU Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   microk8s helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
   microk8s helm repo update
   microk8s helm install gpu-operator \
     -n gpu-operator \
     --create-namespace nvidia/gpu-operator \
     --set toolkit.env[0].name=CONTAINERD_CONFIG \
     --set toolkit.env[0].value=/var/snap/microk8s/current/args/containerd-template.toml \
     --set toolkit.env[1].name=CONTAINERD_SOCKET \
     --set toolkit.env[1].value=/var/snap/microk8s/common/run/containerd.sock \
     --set toolkit.env[2].name=CONTAINERD_RUNTIME_CLASS \
     --set toolkit.env[2].value=nvidia \
     --set toolkit.env[3].name=CONTAINERD_SET_AS_DEFAULT \
     --set-string toolkit.env[3].value=true

.. note::

   If you have configured the GPU operator to use volume mounts for ``DEVICE_LIST_STRATEGY`` on the device plugin and disabled ``ACCEPT_NVIDIA_VISIBLE_DEVICES_ENVVAR_WHEN_UNPRIVILEGED`` on the toolkit, this configuration is currently unsupported, as there is no method to ensure the assigned GPU resource is consistently shared between containers of the same pod.

Verifying Installation
~~~~~~~~~~~~~~~~~~~~~~

Run the following command to verify that all pods are running correctly:

.. code:: bash

   microk8s kubectl get pods -n gpu-operator

You should see output similar to:

.. code:: text

   NAMESPACE          NAME                                                        READY   STATUS      RESTARTS   AGE
   gpu-operator       gpu-operator-node-feature-discovery-gc-76dc6664b8-npkdg       1/1     Running     0          77m
   gpu-operator       gpu-operator-node-feature-discovery-master-7d6b448f6d-76fqj   1/1     Running     0          77m
   gpu-operator       gpu-operator-node-feature-discovery-worker-8wr4n              1/1     Running     0          77m
   gpu-operator       gpu-operator-86656466d6-wjqf4                                 1/1     Running     0          77m
   gpu-operator       nvidia-container-toolkit-daemonset-qffh6                      1/1     Running     0          77m
   gpu-operator       nvidia-dcgm-exporter-vcxsf                                    1/1     Running     0          77m
   gpu-operator       nvidia-cuda-validator-x9qn4                                   0/1     Completed   0          76m
   gpu-operator       nvidia-device-plugin-daemonset-t4j4k                          1/1     Running     0          77m
   gpu-operator       gpu-feature-discovery-8dms9                                   1/1     Running     0          77m
   gpu-operator       nvidia-operator-validator-gjs9m                               1/1     Running     0          77m

Once all pods are running, you can proceed to the :ref:`k8s-installation` section.
