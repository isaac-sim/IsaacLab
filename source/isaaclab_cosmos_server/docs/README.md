# Isaac Lab: Cosmos server

The `isaaclab-cosmos-server` command serves Cosmos3-Nano-Sim-Transfer to Isaac Lab cameras. It runs in the public
Cosmos Framework's own environment, which has its own Python, Torch and CUDA, so it does not import Isaac Lab.
The camera client, `isaaclab_contrib.sensors.cosmos_transfer`, connects to it through `isaaclab_neural_bridge`, over a Unix
socket or TCP.

- `framework.py`: checkpoint loading and inference through Cosmos Framework, and the checks of Cosmos's settings.
- `worker.py`: the `isaaclab-cosmos-server` command, which serves the model with the neural bridge.

Installing and starting the server is described in the
[Cosmos Transfer camera integration guide](../../isaaclab_contrib/docs/cosmos_transfer.md#serve-cosmos).
