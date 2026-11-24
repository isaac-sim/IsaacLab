# Isaac Lab Neural Robot Dynamics

`NeRDCfg` defines the model architecture and physical inputs. Task-specific definitions such as
`AnymalCNeRDCfg` do not contain recording, optimization or inference workflows.

`NeRDTrainingCfg` independently configures optimization. Recording uses task-declared environment
configurations, exploration actions, reset/termination events and existing RecorderManager terms.
The native `TrajectoryWriter` stores captured fields without owning simulation stepping.

`NeRDSolverCfg` and `MJWarpNeRDSolverCfg` configure runtime inference. `NewtonNeRDManager` loads the
checkpoint once and supplies its resources to the adapters through Newton's construction boundary.
The backend owns state, contacts, graph capture and queued model changes.

The scripts in `scripts/neural` are thin launchers for the package-owned entrypoints. Dataset and
checkpoint paths are explicit. Evaluation follows standard RSL-RL play configuration and evaluates
one selected checkpoint per invocation, with training physics separate from evaluation physics.
