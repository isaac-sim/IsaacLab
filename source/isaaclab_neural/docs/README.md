# Isaac Lab Neural Robot Dynamics

`NeRDCfg` defines the model architecture and physical inputs. Task-specific definitions such as
`AnymalCNeRDCfg` do not contain recording, optimization or inference workflows.

`NeRDTrainingCfg` independently configures optimization. Recording uses task-declared environment
configurations, exploration actions, reset/termination events and existing RecorderManager terms.
The recorder term writes batched shards through the native `TrajectoryWriter` and describes the
recorded robot in the dataset. Training builds the model definition on that description.

`NeRDSolverCfg` and `MJWarpNeRDSolverCfg` configure runtime inference. Each solver adapter loads its
checkpoint during normal Newton solver construction and checks it against the live robot.
The backend owns state, contacts, graph capture and queued model changes.

The scripts in `scripts/neural` compose a task with these pieces to record and fit a model; the
package itself imports no task registry. Dataset and checkpoint paths are explicit. Policy training
and playback use the existing RL entrypoints, `isaaclab train` and `isaaclab play`, with `physics=`
selecting the physics preset.
