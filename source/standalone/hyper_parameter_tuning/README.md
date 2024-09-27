# Some utility scripts for hyperparameter tuning.

# Usage
This depeneds on ray tune.

### Cloud
On your cloud provider of choice, configure the following
	- An container registry (GCS artifact registry, AWS ECR, etc)
	- A storage (GCS bucket, AWS S3 bucket, etc)
	- kubernetes provisioning permission
	
### Local
Spin up a ray cluster

