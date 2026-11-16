# Experimental real backend: basic fake session

This is a fake-only experiment for learning what a future Python deployment API
needs. It supports one robot in one environment, joint position commands, and joint telemetry.
It runs on CPU without a viewer, physics solver, or spawned robot USD. The framework still
creates a USD stage and depends on USD and Warp. There is no hardware connection, C++
IsaacDeploy integration, or real-time guarantee.

Run from the repository's uv-managed environment:

```bash
uv run python -m isaaclab_experimental.real.example --steps 30
```

The example uses unchanged `ManagerBasedEnv`, stock `JointPositionAction`, and stock
`mdp.joint_pos_rel` / `mdp.joint_vel_rel` observations. The backend and articulation are
registered through existing extension points; no Isaac Lab core changes are required.
Default reset events are disabled because they teleport simulation state. Episode reset
discards unsubmitted Lab targets and reseeds the Lab command buffer with measured positions
until the next action. It does not cancel a target already accepted by the session: a bare
backend tick continues moving `FakeSession` toward that accepted target. Policy reset does
not stop or home the robot, or reset session time. Cancellation and hold require an explicit
runtime API that this prototype does not provide.

The action exercises joint ordering, scaling, and the default position offset together:

| Quantity | Order | Value |
| --- | --- | --- |
| Initial position and Lab defaults | shoulder, elbow | `[0.1, -0.2]` |
| Policy action | elbow, shoulder | `[0.6, -0.8]` |
| Position target after scale `0.5` and default offset | shoulder, elbow | `[-0.3, 0.1]` |
| Measured position after 30 environment steps | shoulder, elbow | `[-0.3, 0.1]` |

Values are approximate floating-point values. Each environment step advances two runtime
ticks of `0.02` s, so the final session timestamp is `1.2` s. `FakeSession` moves toward
buffered targets with a speed bound of `0.5` per second; writing a command does not replace
measured telemetry. The script prints the measured state, target, observations, and results
of probes for unsupported callers.

The proposed Python boundary is implemented experimentally in [session.py](session.py):

```python
class DeploymentSession(Protocol):
    joint_names: tuple[str, ...]
    def read(self) -> JointState: ...
    def write(self, position_targets: torch.Tensor) -> None: ...
    def step(self, dt: float) -> None: ...
    def close(self) -> None: ...
```

`JointState` contains position and velocity tensors of shape `(num_joints,)` on CPU and a
timestamp in seconds. Names define their order. Positions use meters or radians depending
on joint type. Read tensors are borrowed until the next runtime call; callers must not
modify them. `write` must consume or copy targets before returning. The session owns its
clock and tick; the fake advances logical time without waiting for wall time.

The Lab adapter adds stable batched `ProxyArray` buffers of shape `(1, num_joints)`, default
joint positions and velocities, name selection, and a pending position-command buffer.
It is a plain class: inheriting the full `BaseArticulation` and `BaseArticulationData`
contracts would require implementing 93 and 74 abstract members before learning which
ones deployment actually needs.

Four actual probes expose the current boundary:

| Caller | Missing interface |
| --- | --- |
| `scene.get_state()` | `data.root_pose_w` |
| `mdp.base_lin_vel(env)` | `data.root_lin_vel_b` |
| Joint-state teleport | `write_joint_position_to_sim` |
| Velocity commands | `set_joint_velocity_target_index` |

Unsupported interfaces are deliberately absent and fail at the requesting caller. Passing
the joint-only example does not establish support for arbitrary environments, recording,
floating-base observations, or other command modes.

These findings identify further API work beyond the minimal contract above:

- Activation, connection, passive readiness, and shutdown lifecycle for a physical robot.
- Production ownership and clock semantics, telemetry freshness and validity, and command deadlines.
- A capability descriptor per robot so callers can check supported state and command modes.
- Metadata, defaults, calibration, and mappings between policy and robot joint conventions.
- Optional root/body telemetry, kinematics, and cameras for richer observations and recording.
- Command cancellation and hold, with physical reset and homing separate from policy reset.

These remain questions for a future deployment API; the fake session does not supply these capabilities.
