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

2. Build the Docker container. This may take a while (10-15 minutes). The `--files` flag adds an overlay that bind-mounts the eigenbot extension into the container.
```
python isaaclab/docker/container.py start --files eigenbot.yaml
```

3. Enter the Docker container. This will serve as the main development environment.
```
python isaaclab/docker/container.py enter --files eigenbot.yaml
```

4. Inside the container, install the eigenbot extension and run it.
```bash
# Install the eigenbot extension (one-time, or after rebuilding the container)
cd /workspace/eigenbot
pip install -e source/eigenbot

# Render the eigenbot with zero actions
cd /workspace/eigenbot
python scripts/zero_agent.py --task Template-Eigenbot-Direct-v0 --num_envs 1
```

## Visualization

**X11 forwarding (Linux with display):** The `container.py` script auto-detects X11 and sets it up. The Isaac Sim GUI window should appear on your host display.

**Headless / remote server:** Use Isaac Sim's livestream mode and connect with the Omniverse Streaming Client:
```bash
python scripts/zero_agent.py --task Template-Eigenbot-Direct-v0 --num_envs 1 --livestream 1
```

**WSL2 on Windows:** WSLg handles display forwarding automatically. Run the above commands from within WSL2.