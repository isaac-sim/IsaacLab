* Removed the ``blur`` Cosmos control and ``blur_processor``, which no workflow used: the current Sim-Transfer
  checkpoint is depth-trained, and blur was the only control the service had to copy between GPU and CPU.
