* Added ``isaaclab_experimental.cosmos`` camera integration with control recipes
  and an endpoint client in ``cosmos.client``. Supported synchronous image
  generation for one camera view and retained model weights across episode resets.
* Added a standalone ``isaaclab-cosmos`` package and ``isaaclab-cosmos-server``
  command, with model serving in ``cosmos.server``. Supported public Cosmos
  Framework with compatible Sim-Transfer checkpoints, without installing the
  full Isaac Lab packages in the Framework environment.
* Added service setup instructions and a separate camera integration guide.
