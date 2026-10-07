* Changed the H2 + Sharpa scene assets to use ``ISAACLAB_NUCLEUS_DIR`` directly, replacing the
  temporary ``isaaclab_tasks.contrib.rlinf_assets`` helper. For local mirrors, replace
  ``ISAACLAB_RLINF_DEMO_ASSET_ROOT`` with ``ISAACSIM_ASSET_ROOT`` and retain the
  ``Isaac/IsaacLab`` subtree below that root.
* Changed the apple task's round plate to the uploaded rectangular
  ``Objects/Plate001/plate001.usd`` asset. Reevaluate existing apple-task policies with the updated
  scene before comparing results or resuming training.
