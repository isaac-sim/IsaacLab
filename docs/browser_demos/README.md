<!--
Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Browser simulations

`export.py` is the source for the interactive examples in
[`browser_simulations.rst`](../source/concepts/browser_simulations.rst). Each demo is an
independent manifest, WebAssembly module, and optional policy and visual files. The shared
documentation widget lazy loads the bundle when it enters the viewport. To add
another physics example, export another bundle and add a view in `browser-demo.js`. The
3D views use a locally vendored Three.js module, loaded only when needed.
The VBD, rigid friction, and Cartpole widgets are also embedded beside the corresponding
guidance in `tune_vbd.rst` and `tune_mjwarp.rst`.
All 3D views use the albedo and roughness textures from Isaac Lab's
[`GroundPlaneCfg`](../../source/isaaclab/isaaclab/sim/spawners/from_files/from_files_cfg.py)
default USD asset. Its 2 m repeat gives 1 m checker cells and green landmarks
at each tile corner. The browser's Newton collision plane remains separate from
the visual texture.

The exporter uses Newton 1.6.0, Warp 1.17.0, MuJoCo Warp 3.12.0, and the
[`newton-web`](https://gitlab-master.nvidia.com/lgulich/newton-web) export/compiler
at commit `b0795fbe6b46e08a1fea2425699421415ca4cdf1` (an NVIDIA internal build
tool). It requires Emscripten 5.0.3. The compiled assets are checked into
`docs/source/_static/browser_demos/` so ordinary Sphinx builds do not need the
compiler, the robot USD, or a checkpoint.
Bundles larger than 2 MB are stored as gzip files. The shared browser widget
decompresses them with `DecompressionStream` before initializing WebAssembly.
The gallery presents stiffness, damping, and gravity controls in one VBD widget
and one WebAssembly instance. The VBD guide embeds the same bundle in its own
widget. Locomotion robots share the policy evaluator, joystick controls, and
mesh viewer. Each robot still needs its own reviewed USD, checkpoint, and
exported graph.
The cloth demo adapts the three-value bend-stiffness comparison in
`deformables.rst`. Three free sheets fall across pairs of horizontal rollers;
the outer sheets retain 0.001 and 10 N·m bending stiffness. The logarithmic
slider writes the middle sheet's `edge_bending_properties` before each captured
VBD step, and gravity is also live. The rollers remain in the exported physics
graph; the manifest carries their geometry for browser rendering because
`newton-web` currently serializes only boxes and a plane. The rigid friction
demo draws a solid ramp aligned to its inclined Newton plane and leaves the
default checker ground visible below.
Three MJWarp boxes begin at rest on the ramp; the middle slider writes its
MuJoCo geom friction. The incline has low friction so each box's coefficient
determines the comparison. Both bundles are under 1 MB of WebAssembly and
need no robot asset.
The ground textures are checked into `docs/source/_static/browser_demos/shared/`
from the same hosted asset selected by `GroundPlaneCfg`. The Three.js r170 module
is minified with Terser 5.44.1 and checked into `docs/source/_static/vendor/`
with its MIT license.

## Rebuild

From the Isaac Lab root, install `newton-web` into the uv environment and use
`uv run --no-sync` for the following commands so uv does not remove that
build-only package:

```bash
uv sync --frozen
git clone https://gitlab-master.nvidia.com/lgulich/newton-web /tmp/newton-web
git -C /tmp/newton-web checkout b0795fbe6b46e08a1fea2425699421415ca4cdf1
uv pip install --python .venv/bin/python -e /tmp/newton-web
```

Install Emscripten 5.0.3 using its
[`emsdk` instructions](https://emscripten.org/docs/getting_started/downloads.html).
Then build the stiffness example:

```bash
uv run --no-sync python docs/browser_demos/export.py stiffness \
    --output /tmp/isaaclab-browser-build --emxx /path/to/emsdk/upstream/emscripten/em++
uv run --no-sync python docs/browser_demos/export.py cloth_bending \
    --output /tmp/isaaclab-browser-build --emxx /path/to/emsdk/upstream/emscripten/em++
uv run --no-sync python docs/browser_demos/export.py rigid_friction \
    --output /tmp/isaaclab-browser-build --emxx /path/to/emsdk/upstream/emscripten/em++
```

The Cartpole source is `CARTPOLE_CFG.spawn.usd_path`. Its USD refers to an
instanceable mesh file in the sibling `Props` directory. It uses the
`Isaac-Cartpole-Direct` MJWarp preset, the task's 120 Hz timestep, a fixed
0.2 rad initial pole angle, and the published Newton MJWarp RSL-RL policy.
The policy observes cart position, pole angle, cart velocity, and pole velocity
every two physics steps. The user can temporarily add up to 300 N of cart force;
large sustained pushes can take the cart past its 3 m reset limit.
The viewer uses the USD's collision boxes and adds visual rail supports; the
simulation itself uses only the task asset's collision geometry.

Build it with:

```bash
mkdir -p /tmp/isaaclab-cartpole-source/Props
curl -fL -o /tmp/isaaclab-cartpole-source/cartpole.usd \
    https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/IsaacLab/Robots/Classic/Cartpole/cartpole.usd
curl -fL -o /tmp/isaaclab-cartpole-source/Props/instanceable_meshes.usd \
    https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/IsaacLab/Robots/Classic/Cartpole/Props/instanceable_meshes.usd
curl -fL -o /tmp/isaaclab-cartpole-direct-newton.pt \
    https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Cartpole-Direct_newtonmjwarp_none_rsl_rl.pt
uv run --no-sync python docs/browser_demos/export.py cartpole \
    --usd /tmp/isaaclab-cartpole-source/cartpole.usd \
    --checkpoint /tmp/isaaclab-cartpole-direct-newton.pt \
    --output /tmp/isaaclab-browser-build --emxx /path/to/emsdk/upstream/emscripten/em++
```

The G1 source is `G1_MINIMAL_CFG.spawn.usd_path`, currently the
[`g1_minimal.usd`](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/IsaacLab/Robots/Unitree/G1/g1_minimal.usd)
asset. The policy is the published RSL-RL checkpoint for
`Isaac-Velocity-Flat-G1` with Newton MJWarp and no preset. The visual meshes
come from the matching 37-joint G1 description in
[`ManiSkill`](https://github.com/mani-skill/ManiSkill/tree/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/mani_skill/assets/robots/g1_humanoid),
under Unitree's BSD 3-Clause license:

```bash
curl -fL -o /tmp/g1_minimal.usd \
    https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/IsaacLab/Robots/Unitree/G1/g1_minimal.usd
curl -fL -o /tmp/g1_flat_newton.pt \
    https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-G1_newtonmjwarp_none_rsl_rl.pt
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/mani-skill/ManiSkill.git /tmp/mani-skill-g1
git -C /tmp/mani-skill-g1 fetch --depth 1 origin 62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3
git -C /tmp/mani-skill-g1 checkout 62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3
git -C /tmp/mani-skill-g1 sparse-checkout set mani_skill/assets/robots/g1_humanoid
sha256sum /tmp/g1_minimal.usd /tmp/g1_flat_newton.pt
uv run --no-sync python docs/browser_demos/export.py g1 \
    --usd /tmp/g1_minimal.usd --checkpoint /tmp/g1_flat_newton.pt \
    --visual-source /tmp/mani-skill-g1/mani_skill/assets/robots/g1_humanoid \
    --output /tmp/isaaclab-browser-build --emxx /path/to/emsdk/upstream/emscripten/em++
```

The ANYmal-D source is `ANYMAL_D_CFG.spawn.usd_path`. Its instanceable meshes
are a referenced USD file in a sibling `Props` directory. The policy is the
published RSL-RL checkpoint for `Isaac-Velocity-Flat-AnymalD` with Newton MJWarp
and no preset:

```bash
mkdir -p /tmp/isaaclab-anymal-source/Props
curl -fL -o /tmp/isaaclab-anymal-source/anymal_d.usd \
    https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/IsaacLab/Robots/ANYbotics/ANYmal-D/anymal_d.usd
curl -fL -o /tmp/isaaclab-anymal-source/Props/instanceable_meshes.usd \
    https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/IsaacLab/Robots/ANYbotics/ANYmal-D/Props/instanceable_meshes.usd
curl -fL -o /tmp/isaaclab-anymal-flat.pt \
    https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-AnymalD_newtonmjwarp_none_rsl_rl.pt
uv run --no-sync python docs/browser_demos/export.py anymal \
    --usd /tmp/isaaclab-anymal-source/anymal_d.usd \
    --checkpoint /tmp/isaaclab-anymal-flat.pt \
    --output /tmp/isaaclab-browser-build --emxx /path/to/emsdk/upstream/emscripten/em++
```

Expected input SHA-256 values for this build:

| Input | SHA-256 |
| --- | --- |
| Cartpole USD | `c98ce5dbb174876998052d486036fb79e07f52320851526a4ff61c60ed2db043` |
| Cartpole instanceable meshes | `27976d05b7ee47d7674ab540b8b692113c0052b36f76a5fe9deb01aadc392aa1` |
| Cartpole Newton checkpoint | `251c836e5b6fb9b229ec5e542b7a9071ec49e2da3ebeb3dc230cc77259c76eef` |
| G1 minimal USD | `9dfe7a710aa791e49abf2d9ea74ad3163e291f02f21f59bda9bfcc40f3fab428` |
| G1 flat Newton checkpoint | `3436a12f1f5f6ab51ae2f0c4e386656833bc7df6f1eed10986fef50606fbadaf` |
| G1 visual URDF | `d446ad17340485f694ea17746fcb6e5f09a423cf14aa38efefc8aaefb6d07353` |
| ANYmal-D USD | `8b756c3690808b3b6a9a3fadc62ab843788c7b7835bbe085f0145f173521dc74` |
| ANYmal-D instanceable meshes | `a864b5b9e192592595490f4116319476090f6789830057854c6532020dfc3d33` |
| ANYmal-D flat Newton checkpoint | `0654295241696cdc7855f517a8d94a4951a243f6b21d73152225162ea01aeaaa` |

Copy the contents of `stiffness-web/`, `cloth_bending-web/`, `rigid_friction-web/`, `cartpole-web/`, `g1-web/`, and `anymal-web/` into the corresponding
`docs/source/_static/browser_demos/` directories. Verify the widgets through
an HTTP server, since `file://` URLs cannot load the WebAssembly modules. The
G1 bundle contains compiled Newton code, the actor weights, and decimated
visual meshes in one 1.9 MB binary with the required Unitree license. It does
not copy the robot's USD. The ANYmal-D visual geometry is packed into a 752 KB
binary with its BSD 3-Clause license. Visual meshes are separate from the
exported collision graph.

## Policy contract

The G1 task has 37 actions and 123 policy observations; ANYmal-D has 12 actions
and 48 observations. The widget forms each
observation in the task's order: root linear and angular velocity in the body
frame, projected gravity, a three-value velocity command, joint position
relative to the default pose, joint velocity, and previous action. The actor
uses four dense layers with ELU activations. Position targets are the default
pose plus `0.5 × action`. Control runs every four 5 ms physics steps. The
exporter checks the USD joint count and checkpoint tensor shapes so an asset
or policy swap fails visibly instead of silently driving the wrong model.

The browser uses the G1 link meshes and a convex approximation of the minimal
USD's three collision meshes. It runs MJWarp with two solver iterations instead
of the task's 100, so its dynamics can differ from the full Isaac Lab task. Use
`isaaclab play` for policy evaluation.

ANYmal-D uses its USD collision shapes, 8 browser solver iterations, the task's
contact material and two substeps per 5 ms physics step, and the published actor weights.
The Isaac Lab task uses an LSTM ANYdrive actuator net;
the browser export uses a tuned position PD drive (200 Nm/rad stiffness,
20 Nms/rad damping) and room for 80 contacts. This is an actuator approximation, so the browser is not
a policy validation run. The joystick commands are in the robot frame and
return to zero when released.
