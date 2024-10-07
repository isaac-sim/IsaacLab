#TODO: Move me into docs ;p

# Utility for Distributed Hyperparameter Tuning

This tool utilizes Isaac Lab's built in Hydra configuration, and Ray's built in support
to enable hyperparameter tuning on Kubernetes clusters.


# Installation

This guide includes additional dependencies that are not part of the default Isaac Lab install
as this functionality is still largely experimental.

***You need to install `kubectl`***, which can be done from [this link here](https://kubernetes.io/docs/tasks/tools/)

To install all Python dependencies, run

```
./isaaclab.sh -p -m pip install "ray[default,tune]"
```

### Cloud Setup

On your cloud provider of choice, configure the following

- An container registry (NGC, GCS artifact registry, AWS ECR, etc) where you have
	an Isaac Lab image that you can pull with the correct permissions, configured to 
	support Ray and nvidia-smi
	- See ```../cluster_configs/Dockerfile``` to see how to modify the ```isaac-lab-base```
		container for Ray compatibility. Ray should use the isaac sim python shebang, and nvidia-smi
		should work within the container. Be careful with the setup here as 
		paths need to be configured correctly for everything to work. It's likely that
		the example dockerfile will work for you out of the box.
- A kubernetes Cluster with a GPU-passthrough enabled node-pool with available
	GPU pods that has access to your container registry/storage (likely has to be on same region/VPC), 
	and has the Ray operator enabled with correct IAM permissions. 
- A ``kuberay.yaml.ninja`` file that describes how to allocate resources (already included for
	google cloud)

An example of what a cloud deploy might look look like is in ../cloud_cluster_configs/google_cloud


### Local Setup
Spin up a local kubernetes cluster with GPU passthrough.
For an example, see https://docs.wandb.ai/tutorials/minikube_gpu/

Install ray operator on your local cluster, and make a container where ray is installed. 
See ``../cluster_configs/Dockerfile`` for an example.


# Shared Steps Between Cloud and Local
As a result of using Ray, running experiments in the cloud and locally have very similar steps.

1. Start your kubernetes server and verify you have access

	``kubectl get nodes`` should list your nodes

2. Ensure Ray Operator is installed on your cluster

	``kubectl get crds | grep ray`` should list rayclusters.ray.io , rayjobs.ray.io , and
	rayservices.ray.io

2. Spin up your KubeRay integration from the template cluster configuration file

	See ``./isaaclab.sh -p source/standalone/workflows/ray/hyper_parameter_tuning/launch.py -h``
	for all possible arguments

	For example, you could invoke with: 

	``
	./isaaclab.sh -p source/standalone/workflows/ray/hyper_parameter_tuning/launch.py 
	 --cluster_host google_cloud --namespace <NAMESPACE>  --image <CUSTOM_ISAAC_RAY_IMAGE> --min_workers 4 --max_workers 16
	 ``

3. Check that your KubeRay cluster worked with `kubectl get pods` and `kubectl describe pods`.
	It may take a few minutes for the cluster to spin up. If there is an error, or a crash loop backup,
	you can inspect the logs further with ``kubectl logs <POD_NAME>``. When all pods
	say ``Running`` as their status, the cluster is ready to tune hyperparameters.

4. Check that you can issue jobs to the cluster, that all GPUs are available,
	that Ray is installed correctly, nvidia-smi works, and that the needed deps
	for Ray/Isaac Lab are found on path with the following command:

	``
	./source/standalone/workflows/ray/hyper_parameter_tuning/grok_ray_address.sh &&
	ray job submit --working-dir source/standalone/workflows/ray/hyper_parameter_tuning \
	--address=$(cat ~/.ray_address) -- /workspace/isaaclab/_isaac_sim/python.sh \
	test_kuberay_config.py --num_jobs 4
	``


4. Define your desired hyperparameter tuning configuration in a .py file on your local host.
	For an example, see ``source/standalone/workflows/ray/hyper_parameter_tuning/config/cartpole_sweep.py``

5. Start your hyperparameter tune sweep job with 

	``
	./source/standalone/workflows/ray/hyper_parameter_tuning/grok_ray_address.sh &&
	ray job submit --working-dir source/standalone/workflows/ray/hyper_parameter_tuning \
	--address=$(cat ~/.ray_address) -- /workspace/isaaclab/_isaac_sim/python.sh \
	<YOUR_JOB_HERE>
	``

	For example,

	``
	./source/standalone/workflows/ray/hyper_parameter_tuning/grok_ray_address.sh &&
	ray job submit --working-dir source/standalone/workflows/ray/hyper_parameter_tuning \
	--address=$(cat ~/.ray_address) -- /workspace/isaaclab/_isaac_sim/python.sh \
	config/cartpole_sweep.py 
	``
6. When you have completed your hyperparameter tune sweep job, stop the cluster
	``kubectl delete raycluster <CLUSTER_NAME> -n <NAMESPACE>``


##
Notes
https://discuss.ray.io/t/how-to-define-fcnet-hiddens-size-and-number-of-layers-in-rllib-tune/6504/18
