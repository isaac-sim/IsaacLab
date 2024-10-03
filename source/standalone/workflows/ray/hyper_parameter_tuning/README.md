#TODO: Move me into docs ;p

# Utility for Distributed Hyperparameter Tuning

This tool utilizes Isaac Lab's built in Hydra configuration, and Ray's built in support
to enable hyperparameter tuning on Kubernetes clusters.


# Installation

This guide includes additional dependencies that are not part of the default Isaac Lab install
as this functionality is still largely experimental.

***You also need to install `kubectl`***, which can be done from [this link here](https://kubernetes.io/docs/tasks/tools/)

To install all Python dependencies, run

```
./isaaclab.sh -p -m pip install "ray[tune]"
```

### Cloud Setup

On your cloud provider of choice, configure the following

- An container registry (NGC, GCS artifact registry, AWS ECR, etc) where you have
	an Isaac Lab image that you have pull with the correct permissions
- A storage (GCS bucket, AWS S3 bucket, etc)
- A kubernetes Cluster with a GPU-passthrough enabled node-pool that has access to
	your container registry/storage (likely has to be on same region/VPC), and has the Ray operator enabled
	with correct IAM permissions
- A ``kuberay.yaml.ninja`` file that describes how to allocate resources (example already included for
	google cloud)

An example of what a cloud deploy might look look like is in ../cloud_cluster_configs/google_cloud


### Local Setup
Spin up a local kubernetes cluster with GPU passthrough.
For an example, see https://docs.wandb.ai/tutorials/minikube_gpu/

Install ray operator on your local cluster.


# Shared Steps Between Cloud and Local
As a result of using Ray, running experiments in the cloud and locally have very similar steps

1. Start your kubernetes server and verify you have access

	``kubectl get nodes`` should list your nodes in your node pool

2. Ensure Ray Operator is installed on your cluster

	``kubectl get crds | grep ray`` should list rayclusters.ray.io , rayjobs.ray.io , and
	rayservices.ray.io

2. Spin up your KubeRay integration from the template cluster configuration file

	``./isaaclab.sh -p source/workflows/ray/hyper_parameter_tuning``

3. Check that your KubeRay integration worked with `kubectl get pods` and `kubectl describe pods`

4. Define your desired hyperparameter sweep in a .py file on your local host.
	For an example, see ``source/standalone/workflows/ray/hyper_parameter_tuning/config/cartpole_sweep.py``

5. Start your hyperparameter tune sweep job

6. When you have completed your hyperparameter tune sweep job, stop the cluster
	``kubectl delete raycluster <CLUSTER_NAME> -n <NAMESPACE>``



###
Notes
https://discuss.ray.io/t/how-to-define-fcnet-hiddens-size-and-number-of-layers-in-rllib-tune/6504/18
