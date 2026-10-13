# GB300 RJ45 insertion and extraction

Non-RL manager-based environment with a Rizon 4s / Sharpa hand, two sockets,
a spring-clip connector and a VBD cable. One end is fixed; the loose end starts
physically grasped. The example inserts, releases, regrasps, presses the clip and
extracts in one continuous simulation.
The checked-in scene is the minimal physical workcell. The full GB300 facility
and cinematic overlays remain separate local presentation tooling.

```bash
uv run isaaclab demo rizon-sharpa-rj45
uv run --extra isaacsim --extra video isaaclab demo rizon-sharpa-rj45 --viz kit --video
```

The example supplies absolute palm/finger targets through Isaac Lab's Newton IK
and action manager. Native hydroelastic hand–housing/clip contact is enabled;
the VBD endpoint follows the measured plug without cable force feedback.
There are no rewards, training presets, policy checkpoints or automatic episode resets.

## Requirements

- Robot: pinned Fabrics-Sim USD and textures; set
  `ISAACLAB_FABRICS_SIM_RIZON_SHARPA_ROOT`. Revision and bundle checks are in `assets.py`.
  This access-controlled default must be replaced/validated before public release.
- Connector: Newton's canonical `rj45_plug.usd`; set `ISAACLAB_RJ45_PLUG_USD`
  if it is absent from the installation. The resolver reports its pinned download URL.
- The prerequisite [Newton dependency update](https://github.com/isaac-sim/IsaacLab/pull/8524) includes the merged
  [contact-gain fix](https://github.com/newton-physics/newton/pull/4649) and matching
  MuJoCo/MJWarp 3.15. No private Newton patch or `PYTHONPATH` override is needed.

`--video` uses the standard recorder. `--failure_tail_s 8` keeps recording after
a controller fault without bypassing its guards. No robot/plug teleport, grasp
attachment or handoff reset is used.
