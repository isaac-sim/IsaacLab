* Added the ``isaaclab_cosmos_server`` package with the ``isaaclab-cosmos-server`` command. It serves the public
  Cosmos Framework from its own environment, keeps the model loaded between sessions, and generates all of a
  session's views as one batch. Images move over a Unix socket or TCP, in the messages of
  :mod:`isaaclab_cosmos_server.protocol`, which the Isaac Lab camera client shares.
