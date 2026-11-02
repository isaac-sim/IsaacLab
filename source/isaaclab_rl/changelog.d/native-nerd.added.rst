* Added offline NeRD fitting from native HDF5 episodes, with bounded history sampling, held-out
  episode validation, atomic resumable checkpoints and optional synchronous W&B model uploads.
* Supported ParallelNeRD/VectorNeRD, mixed precision with resumable gradient scaling and distributed
  fitting with global batches, partitioned validation and per-rank sampler checkpoints.
* Added offline NeRD contact-set fitting with valid-geometry normalization and preserved categorical identities.

* Shared PhysicsNeMo field normalization between fitting and deployment. Version-3 checkpoints require
  refitting existing raw recordings instead of silently reinterpreting earlier heading-normalized weights.
