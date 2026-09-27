Fixed
^^^^^

* Preserved Newton actuator PID and delay history across graph replays with odd decimation by
  copying the final actuator state back into the graph's input buffers.
