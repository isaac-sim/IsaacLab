* Added the ``isaaclab_cosmos_server`` package with the ``isaaclab-cosmos-server`` command. It serves the public
  Cosmos Framework from its own environment through ``isaaclab_neural_bridge``, keeps the model loaded between
  sessions, and generates all of a session's views as one batch. It checks the camera client's Cosmos settings and
  announces its chunk sizes and canvases. On one Linux machine, images stay on the GPU through the bridge's CUDA IPC
  between GPUs that can access each other; otherwise they travel in the socket messages.
