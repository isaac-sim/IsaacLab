* Changed the Cosmos server's generation history to the Sim-Transfer recipe's 30-latent window with 3 attention
  sinks (was 8 and 3), set with ``isaaclab-cosmos-server --kv-window`` and ``--attention-sink``.
* Changed the Cosmos service to use prompts as given, like the Sim-Transfer recipe; it no longer appends an
  instruction to follow the control video.
* Changed the default :attr:`~isaaclab_experimental.cosmos.CosmosModelCfg.modality` from ``"edge"`` to
  ``"depth"``, the control the Shadow Hand preset uses.
