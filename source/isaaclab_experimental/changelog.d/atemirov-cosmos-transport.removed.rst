* Removed the ``blur`` Cosmos control and ``blur_processor``, which no workflow used: the current Sim-Transfer
  checkpoint is depth-trained, and blur was the only control the service had to copy between GPU and CPU.
* Removed the ``edge`` and ``seg`` Cosmos controls with their recipes and helpers (``edge_processor``,
  ``regional_edge_processor``, ``segmentation_processor``, ``edge_control``, ``region_control``,
  ``regional_edges``, ``RegionalEdgeControl``) and ``CosmosModelCfg.modality``: the checkpoint is depth-trained,
  so Cosmos uses depth control only. The service protocol is now version 3.
