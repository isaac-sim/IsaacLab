* Fixed 40 ms stalls of small Cosmos messages over TCP: each message was written in two parts without
  ``TCP_NODELAY``, so it waited for a delayed acknowledgment. Messages are now written at once with
  ``TCP_NODELAY`` on both ends.
* Fixed the Cosmos connection error to name the endpoint and how to start the server.
