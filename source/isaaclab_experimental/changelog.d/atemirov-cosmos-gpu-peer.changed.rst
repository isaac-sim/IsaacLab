* Changed Cosmos CUDA IPC sessions to have each process create the buffer it writes and the event it records on its
  own GPU; the camera no longer creates the output buffer and event. The Cosmos protocol version is now 3, so a
  camera and a service from different versions refuse to connect.
