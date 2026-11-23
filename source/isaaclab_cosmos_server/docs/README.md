# Isaac Lab: Cosmos server

The `isaaclab-cosmos-server` command serves Cosmos3-Nano-Sim-Transfer to Isaac Lab cameras. It runs in the public
Cosmos Framework's own environment, which has its own Python, Torch and CUDA, so it does not import Isaac Lab.
The camera client, `isaaclab_contrib.sensors.cosmos`, connects to it over a Unix socket or TCP.

- `protocol.py`: the socket message format and endpoints, shared with the camera client.
- `framework.py`: checkpoint loading and inference through Cosmos Framework.
- `service.py`: the resident service that keeps the model loaded between camera sessions.
- `worker.py`: the `isaaclab-cosmos-server` command.

Installing and starting the server is described in the
[Cosmos Transfer camera integration guide](../../isaaclab_contrib/docs/cosmos_transfer.md#serve-cosmos).
