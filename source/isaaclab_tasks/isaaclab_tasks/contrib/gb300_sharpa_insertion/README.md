# Sharpa RJ45

Non-RL manager-based environment with a Rizon 4s / Sharpa hand, two sockets,
a spring-clip connector and a VBD cable. One end is fixed; the loose end starts
physically grasped. The example inserts, releases, regrasps, presses the clip and
extracts in one continuous simulation.
NewtonGL shows one full-detail GB300 rack. Kit and Newton RTX compose the full
data-center USD and eight foreground racks through USD-only scene cloning;
this scenery adds no physics bodies or colliders. Cinematic overlays remain local.

```bash
uv run isaaclab example sharpa-rj45
uv run --extra ovrtx isaaclab example sharpa-rj45 --viz newton_rtx
uv run --extra isaacsim --extra video isaaclab example sharpa-rj45 --viz kit --video
```

The example supplies absolute palm/finger targets through Isaac Lab's Newton IK
and action manager. Native hydroelastic hand–housing/clip contact is enabled;
the VBD endpoint follows the measured plug without cable force feedback.
Dynamics and collision detection run at 480 Hz with four VBD iterations. The
native IK solve is CUDA-captured; the scripted state machine still reads measured state on the CPU.
There are no rewards, training presets, policy checkpoints or automatic episode resets.

## Requirements

- Visuals: `Environments/RizonSharpaRJ45` on the standard Isaac Lab asset server. Set
  `ISAACLAB_SHARPA_RJ45_VISUAL_ROOT` to an alternate bundle root.
- Robot: `Robots/Rizon4sSharpa` in the same bundle, loaded through the native USD resolver by default.
  Set `ISAACLAB_FABRICS_SIM_RIZON_SHARPA_ROOT` to a local USD bundle with its complete textures
  for offline use. Explicit local bundles retain the revision and integrity checks in `assets.py`.
- Connector: Newton's canonical `rj45_plug.usd`; set `ISAACLAB_RJ45_PLUG_USD`
  if it is absent from the installation. The resolver reports its pinned download URL.
- The prerequisite [Newton dependency update](https://github.com/isaac-sim/IsaacLab/pull/8524) includes the merged
  [contact-gain fix](https://github.com/newton-physics/newton/pull/4649) and matching
  MuJoCo/MJWarp 3.15. No private Newton patch or `PYTHONPATH` override is needed.

`--video` uses the standard recorder. `--failure_tail_s 8` keeps recording after
a controller fault without bypassing its guards. No robot/plug teleport, grasp
attachment or handoff reset is used.
