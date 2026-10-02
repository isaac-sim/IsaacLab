<!--
Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Browser simulations

`export.py` is the source for the interactive examples in the
[VBD tuning](../source/concepts/solver-tuning/tune_vbd.rst),
[MJWarp tuning](../source/concepts/solver-tuning/tune_mjwarp.rst),
[MPM](../source/concepts/using_mpm.rst),
[actuator](../source/concepts/actuators.rst), and
[reinforcement learning](../source/concepts/reinforcement_learning.rst) guides. Each demo is an
independent manifest, WebAssembly module, and optional policy and visual files. The shared
documentation widget lazy loads the bundle when it enters the viewport. To add
another physics example, export another bundle and add a view in `browser-demo.js`. The
3D views use a locally vendored Three.js module, loaded only when needed.
Each widget appears beside the guidance it illustrates.
Use `.. isaaclab-browser-demo:: <name>` to embed a bundle; the shared Sphinx
directive checks its files and ABI and loads the widget assets once per page.
The [contributor guide](../source/developer-tools/interactive_examples.rst)
describes the capture boundary, current manual adaptations, candidate examples,
native browser sessions, and version maintenance.
All 3D views use the albedo and roughness textures from Isaac Lab's
[`GroundPlaneCfg`](../../source/isaaclab/isaaclab/sim/spawners/from_files/from_files_cfg.py)
default USD asset. Its 2 m repeat gives 1 m checker cells and green landmarks
at each tile corner. The browser's Newton collision plane remains separate from
the visual texture.

The exporter uses Newton 1.6.0, Warp 1.17.0, MuJoCo Warp 3.12.0, and the
internal `newton-web` export/compiler
at commit `b0795fbe6b46e08a1fea2425699421415ca4cdf1` (an NVIDIA internal build
tool). It requires Emscripten 5.0.3. The compiled assets are checked into
`docs/source/_static/browser_demos/` so ordinary Sphinx builds do not need the
compiler or external robot and policy assets.
Bundles larger than 2 MB are stored as gzip files. The shared browser widget
decompresses them with `DecompressionStream` before initializing WebAssembly.
The compressed MPM module also exceeds 2 MB and uses Git LFS; fetch it with
`git lfs pull` before building the documentation. The standalone GitHub preview
selects GitHub's media URL through the widget's optional `wasm-src` attribute,
while documentation builds use the local bundle.
The VBD widget exposes shear, volume, damping, and gravity in one WebAssembly
instance. The joint PD widget exposes the target and both drive gains. Locomotion
robots share the policy evaluator, joystick controls, and
mesh viewer. Each robot still needs its own reviewed asset, policy, and exported graph.
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
determines the comparison. The joint PD demo drives a suspended three-joint
arm. The browser supplies step and sine targets and plots measured joint angles;
the exported graph advances MJWarp with live per-joint implicit-drive gains.
These bundles are under 1 MB of WebAssembly and
need no robot asset.
The MPM demo drops one deterministically jittered 1,728-particle block over a
horizontal cylinder into a 1.9 m × 1.5 m catch tub with a 0.42 m rim.
Clear walls reveal the collected material. It uses a fixed 0.1 m grid and a 60 Hz implicit MPM
step with up to 50 Jacobi iterations. Five controls set `young_modulus`,
`yield_pressure`, `friction`, `yield_stress`, and `hardening` on reset.
Sand, snow, clay, and water buttons select qualitative reference materials,
including Poisson ratio and tensile yield ratio, and restart the drop.
The 12 × 12 × 12 lattice samples the same 0.35 m cube at a density of
1,000 kg/m³. Displayed particle radii are 40% of the lattice spacing;
simulation radii remain 50%, so reducing the displayed grain size does not
change the material volume.

| Reference | Stiffness [Pa] | Compression yield [Pa] | Friction | Cohesion [Pa] | Hardening | Poisson ratio | Tensile yield ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sand | 100,000 | 100,000 | 0.65 | 0 | 0 | 0.3 | 0 |
| Snow | 50,000 | 1,000 | 0.1 | 1,000 | 1.5 | 0.3 | 0.2 |
| Clay | 100,000 | 100,000 | 0.1 | 5,000 | 0 | 0.3 | 0.5 |
| Water | 10,000 | 100,000 | 0 | 0 | 0 | 0.49 | 0 |

These presets demonstrate material response on a coarse grid; they are not
calibrated material models. Water retains the particle view without surface
reconstruction.

Reset reloads the complete captured state, including plastic strain and solver
history, while preserving the selected material. A build-only solver subclass
supplies analytic cylinder, inner tub wall, rim, and floor distances and normals to Newton's grid
contact solver, with Newton's Coulomb response for particle projection, avoiding
process-local mesh handles. For the scalar P0 strain
basis it uses the diagonal cell volumes directly, including Newton's small-volume
cutoff. The export enables Warp's conditional CPU graph so convergence is checked
at replay instead of reading an unfinished capture. It also deduplicates recorded
allocation aliases to preserve the owning capacity when Warp 1.17 serializes
Newton's residual-reduction views. These hooks rely on the pinned Newton and Warp
versions. The browser advances the compiled Newton kernels.
Density is fixed at 1,000 kg/m³ and the elastic damping relaxation time at
0.01 s. Each two-second drop repeats; at most one physics step
runs per animation frame, keeping controls responsive on slower CPUs. Playback
speed depends on the device. The finite grid is intended for this compact comparison.
The viewer renders the 1,728 moving particles with one instanced draw.
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
Then build the physics-only examples:

```bash
uv run --no-sync python docs/browser_demos/export.py --check-toolchain
uv run --no-sync python docs/browser_demos/export.py stiffness \
    --output /tmp/isaaclab-browser-build --emxx /path/to/emsdk/upstream/emscripten/em++
uv run --no-sync python docs/browser_demos/export.py cloth_bending \
    --output /tmp/isaaclab-browser-build --emxx /path/to/emsdk/upstream/emscripten/em++
uv run --no-sync python docs/browser_demos/export.py mpm \
    --output /tmp/isaaclab-browser-build --emxx /path/to/emsdk/upstream/emscripten/em++
uv run --no-sync python docs/browser_demos/export.py rigid_friction \
    --output /tmp/isaaclab-browser-build --emxx /path/to/emsdk/upstream/emscripten/em++
uv run --no-sync python docs/browser_demos/export.py joint_pd \
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

The G1 uses Unitree's BSD 3-Clause licensed
[`g1_29dof_rev_1_0.urdf`](https://github.com/unitreerobotics/unitree_ros/blob/ccfc6fd8430a17ba3dacef9a1e2faf64ff3b0aee/robots/g1_description/g1_29dof_rev_1_0.urdf)
and visual meshes. Its actor is the WBC-AGILE Apache 2.0 licensed
[`Velocity-G1-v0`](https://github.com/nvidia-isaac/WBC-AGILE/tree/v1.3.1/agile/data/policy/velocity_g1/leapp/Velocity-G1-v0)
ONNX model. Fetch the published actor (including Git LFS) and its policy description:

```bash
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/unitreerobotics/unitree_ros.git /tmp/unitree-ros
git -C /tmp/unitree-ros sparse-checkout set robots/g1_description
git -C /tmp/unitree-ros fetch --depth 1 origin ccfc6fd8430a17ba3dacef9a1e2faf64ff3b0aee
git -C /tmp/unitree-ros checkout ccfc6fd8430a17ba3dacef9a1e2faf64ff3b0aee
git clone --branch v1.3.1 --depth 1 --filter=blob:none --sparse \
    https://github.com/nvidia-isaac/WBC-AGILE.git /tmp/wbc-agile
git -C /tmp/wbc-agile sparse-checkout set agile/data/policy/velocity_g1/leapp/Velocity-G1-v0
git -C /tmp/wbc-agile lfs pull
uv run --no-sync --with onnx python docs/browser_demos/export.py g1 \
    --checkpoint /tmp/wbc-agile/agile/data/policy/velocity_g1/leapp/Velocity-G1-v0/Velocity-G1-v0.onnx \
    --policy-description /tmp/wbc-agile/agile/data/policy/velocity_g1/leapp/Velocity-G1-v0/Velocity-G1-v0.yaml \
    --policy-license /tmp/wbc-agile/LICENCE \
    --visual-source /tmp/unitree-ros/robots/g1_description \
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
| Unitree G1 29-joint URDF | `c0ae739c640c3e2c00d1bdd8810b5d6e59601487bd1a3995859f9543269ee5c8` |
| WBC-AGILE G1 ONNX policy | `4c92c5a64d1220ab02b042e77bdd69bcd2c0310590755c3dd53b5bde229d26e4` |
| ANYmal-D USD | `8b756c3690808b3b6a9a3fadc62ab843788c7b7835bbe085f0145f173521dc74` |
| ANYmal-D instanceable meshes | `a864b5b9e192592595490f4116319476090f6789830057854c6532020dfc3d33` |
| ANYmal-D flat Newton checkpoint | `0654295241696cdc7855f517a8d94a4951a243f6b21d73152225162ea01aeaaa` |

Copy the contents of `stiffness-web/`, `cloth_bending-web/`, `mpm-web/`, `rigid_friction-web/`, `joint_pd-web/`, `cartpole-web/`, `g1-web/`, and `anymal-web/` into the corresponding
`docs/source/_static/browser_demos/` directories. Verify the widgets through
an HTTP server, since `file://` URLs cannot load the WebAssembly modules. The
G1 bundle contains compiled Newton code, the actor weights, and decimated
visual meshes in one 1.5 MB binary with the required Unitree and WBC-AGILE licenses.
The ANYmal-D visual geometry is packed into a 752 KB
binary with its BSD 3-Clause license. Visual meshes are separate from the
exported collision graph.

## Policy contract

Published WASM bundles are independent of the Python environment's Warp version.
Rebuilding requires the reviewed Newton/Warp/MJWarp pins and Newton Web checkout;
the exporter rejects mismatches before capture. New builds record these versions
under `manifest.build.isaacLabExporter`. Upgrade the compiler and review the affected
adapters together, then compare native/WASM behavior before updating the pins.

The G1 policy has 83 observations and 12 lower-body actions, mapped by name into
the 29-joint Unitree asset. The observations are command (four values, with the
fourth reserved), body-frame linear and angular velocity, projected gravity,
joint positions relative to the default pose, joint velocities scaled by 0.1,
and previous action. Four dense layers with ELU activations produce joint targets
at 50 Hz. The browser advances MJWarp at 1 kHz with one solver iteration and
the reference's 12 primitive contact shapes. The exporter verifies the source
hashes, joint names, and layer dimensions. Policy and visual licenses ship with
the bundle.

ANYmal-D has 12 actions and 48 observations. Its actor uses four dense layers
with ELU activations. Position targets are the default pose plus `0.5 × action`.
Control runs every four 5 ms physics steps.

ANYmal-D uses its USD collision shapes, 8 browser solver iterations, the task's
contact material and two substeps per 5 ms physics step, and the published actor weights.
The Isaac Lab task uses an LSTM ANYdrive actuator net;
the browser export uses a tuned position PD drive (200 Nm/rad stiffness,
20 Nms/rad damping) and room for 80 contacts. This is an actuator approximation, so the browser is not
a policy validation run. The joystick commands are in the robot frame and
return to zero when released.
