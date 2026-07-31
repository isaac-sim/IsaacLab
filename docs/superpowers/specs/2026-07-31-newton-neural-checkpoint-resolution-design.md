# Newton Neural Checkpoint Resolution Design

## Goal

Enable Newton neural actuators to load actuator-network checkpoints from the
same local and remote paths supported by Isaac Lab's regular neural actuators.

## Design

The Newton actuator schema authoring path currently passes
`ActuatorNetMLPCfg.network_file` and `ActuatorNetLSTMCfg.network_file` directly
to PyTorch while adding Newton metadata. PyTorch accepts local filesystem paths
but not the HTTPS asset URLs used by the Go1 and ANYmal configurations.

Before loading the checkpoint, `_resave_checkpoint_with_metadata()` will call
`isaaclab.utils.assets.retrieve_file_path()`. That existing helper returns an
absolute local path unchanged and downloads remote assets into the system
temporary directory. Its existing cache behavior avoids downloading an asset
again when the cached file already exists.

The resolved local path will be passed to both the TorchScript loader and the
dictionary-checkpoint fallback. Error messages will continue to identify the
original configured path so users can locate the failing asset configuration.
The patched checkpoint containing Newton metadata will continue to be written
to a process-lifetime temporary file and authored into the USD asset path.

## Alternatives Considered

- Resolving the path in `_author_actuator_prims()` would fix the current caller,
  but would leave the checkpoint-resave helper unsafe for any future caller.
- Loading remote bytes through `read_file()` would match the regular actuator
  loader but would not provide the requested persistent temporary-file cache.
- Adding a Newton-specific downloader or cache would duplicate existing Isaac
  Lab behavior and introduce unnecessary maintenance.

## Error Handling

Failures reported by `retrieve_file_path()` will propagate without being
hidden. A downloaded file that is neither a TorchScript archive nor a supported
dictionary checkpoint will retain the existing `ValueError` behavior.

## Verification

No new automated regression test will be added, per maintainer direction. The
change will be validated by:

1. Running the relevant existing formatting and checks.
2. Launching the Go1 flat task with PhysX, Newton actuators, and its remote MLP
   checkpoint.
3. Launching the ANYmal-D flat task with PhysX, Newton actuators, and its remote
   LSTM checkpoint.
4. Confirming both runs pass checkpoint loading and begin training.

## Scope

The change is limited to neural-checkpoint path resolution during Newton
actuator schema authoring. It does not change actuator inference, metadata
contents, task configuration, or the shared asset retrieval implementation.
