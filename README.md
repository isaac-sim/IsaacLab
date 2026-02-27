# Eigenbot IsaacLab
Repository for Eigenbot IsaacLab simulation and RL code, ported from the Eigenbot IsaacGym implementation.

## Environment Setup
The following environment setup steps has been tested on a Linux Ubuntu 20.04.6 LTS system. For newer Linux or Windows systems, alternative IsaacLab installation methods (using pip or pre-installed binary methods) can be used.

Docker containerization bypasses versioning issues, so this is the method described below. Ensure that Docker, Docker Compose, and Nvidia Container Toolkit is installed.

1. Clone the GitHub repository
```
git clone https://github.com/biorobotics/eigenbot_isaaclab
cd eigenbot_isaaclab
```

2. Build the Docker container. This may take a while (10-15 minutes).
```
python isaaclab/docker/container.py start
```

3. Enter the Docker container. This will serve as the main development environment
```
python isaaclab/docker/container.py enter
```