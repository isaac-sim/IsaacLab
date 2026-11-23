# Isaac Lab: neural bridge

Connects Isaac Lab to a model that runs in its own process and Python environment, for example a model that needs
another Torch version. Both processes install this package; it imports neither Isaac Lab nor a model framework. It
is separate from `isaaclab_neural`, the neural backend, because the model's environment installs it without Isaac Lab.

- `protocol.py`: bounded messages, endpoints (`unix:///path` or `tcp://host:port`), and one-shot requests.
- `ipc.py`: GPU buffers and events shared between the two processes on one Linux machine, through Warp's CUDA IPC.
- `client.py`: `BridgeSession`, one generation session from the Isaac Lab side: it connects, chooses CUDA IPC or
  the socket, sends each step's controls and returns the generated images on the controls' device.
- `server.py`: `serve(model, endpoint)`, the resident service loop. It keeps the model loaded between sessions, runs
  it on one inference thread, and passes each session's model settings to the model, which checks them.

A served model has a `capabilities` dictionary, which clients read with `status` (the service reads `device` and
`max_chunk_frames` from it), and `warmup()`, `open_stream(num_views=, seeds=, height=, width=, **settings)` and
`close()`. Its stream has `step(controls, reset_rows, seeds, **episode)` and `close()`.

The Cosmos Transfer model is served through it by `isaaclab_cosmos_server`; see the
[Cosmos Transfer camera integration guide](../../isaaclab_contrib/docs/cosmos_transfer.md).
