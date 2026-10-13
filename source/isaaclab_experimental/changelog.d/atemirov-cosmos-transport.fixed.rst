* Fixed delays of small Cosmos messages over TCP: each message was written in two parts without
  ``TCP_NODELAY``, so it could wait 40 ms for a delayed acknowledgment. Messages are now written at once with
  ``TCP_NODELAY`` on both ends.
* Fixed the Cosmos connection error to name the endpoint and how to start the server.
* Fixed CUDA IPC session setup leaking buffers and events when a later resource could not be created or opened.
