# Snap Circuits Apple Vision Pro demo

This demo starts Apple Vision Pro CloudXR teleoperation with a packing table
populated from three asset sources:

1. `s3://ai-lumalabs-datasets-ap-se-2-lance/Opal_Sim/shared/assets/objects/snap_circuits/`
2. `sc100_mesh_bundle.zip` (21 approximate SC-100 component classes)
3. `test_tube_16mm_rack_18mm_compatible.zip` (rack and tube meshes)

Three hand backends are available:

- `IsaacContrib-PickPlace-GR1T2-SnapCircuits-Abs` (default) uses the original,
  proven GR1 hands and includes both the wheel and curated circuit pieces.

- `IsaacContrib-SnapCircuits-SharpaWave-Abs` (default) uses Sharpa's official
  dual floating-base Wave USD and all 22 finger joints per hand.
- `IsaacContrib-SnapCircuits-ProHand-Abs` uses Proception's left/right
  ProHand Gen1.D USDs and the thumb-first 20-joint SDK finger contract.

Both use IsaacTeleop's DexPilot retargeter. Vision Pro wrist tracking commands
the simulated wrist/palm pose and hand tracking commands the fingers.

The private S3 USDs are the authoritative circuit parts. When both the
original and `_02` revision exist, preparation chooses `_02`; `.bak-*` files
are ignored. The SC-100 ZIP and rack/tube ZIP add their downloaded objects.
All source assets are deliberately excluded from Git. `setup-assets.sh`
produces a local, ignored `assets/` directory. The default bench contains 17
curated objects rather than the 45-object discovered catalog: one S3 base
grid, nine native S3 circuit parts, five non-duplicate SC-100 parts, and the
requested tube and rack.

## 1. Put the private S3 assets on robolab1

Run these commands on a machine that already has the correct rclone config.
Replace `lance:` with the configured remote reported by `rclone listremotes`.
Set the checkout used on robolab1 once; the existing lab deployment uses
`/home/benbiggs/IsaacLab-3`, while a fresh clone may use `~/IsaacLab`.

```bash
export ISAACLAB_ROOT=/home/benbiggs/IsaacLab-3
rclone listremotes
rclone copy \
  'lance:ai-lumalabs-datasets-ap-se-2-lance/Opal_Sim/shared/assets/objects/snap_circuits/' \
  /tmp/isaaclab-snap-circuits-s3 \
  --progress

ssh robolab1 \
  "mkdir -p ${ISAACLAB_ROOT}/docker/apple-vision-pro/snap-circuits/assets/source/s3-snap-circuits"
scp -r /tmp/isaaclab-snap-circuits-s3/. \
  "robolab1:${ISAACLAB_ROOT}/docker/apple-vision-pro/snap-circuits/assets/source/s3-snap-circuits/"
```

If the remote itself is rooted at the bucket, omit the bucket name:

```bash
rclone copy \
  'lance:Opal_Sim/shared/assets/objects/snap_circuits/' \
  /tmp/isaaclab-snap-circuits-s3 \
  --progress
```

Verify the remote path before copying:

```bash
rclone lsf \
  'lance:ai-lumalabs-datasets-ap-se-2-lance/Opal_Sim/shared/assets/objects/snap_circuits/' \
  --max-depth 2
```

If rclone is configured directly on robolab1, skip the local copy and pass its
remote to `setup-assets.sh` in step 4.

## 2. Copy the two downloaded ZIPs

From the Mac:

```bash
export ISAACLAB_ROOT=/home/benbiggs/IsaacLab-3
scp \
  ~/Downloads/sc100_mesh_bundle.zip \
  ~/Downloads/test_tube_16mm_rack_18mm_compatible.zip \
  "robolab1:${ISAACLAB_ROOT}/docker/apple-vision-pro/snap-circuits/assets/source/"
```

The expected files on robolab1 are:

```text
/home/benbiggs/IsaacLab-3/docker/apple-vision-pro/snap-circuits/assets/source/
├── sc100_mesh_bundle.zip
├── test_tube_16mm_rack_18mm_compatible.zip
└── s3-snap-circuits/
```

## 3. Start/recreate the Isaac Lab container

On robolab1:

```bash
ssh robolab1
export ISAACLAB_ROOT=/home/benbiggs/IsaacLab-3
cd "$ISAACLAB_ROOT"

printf '[X11]\nx11_forwarding_enabled = 0\n' > docker/.container.cfg
./docker/container.py start
```

The compose configuration mounts `docker/apple-vision-pro` into
`/workspace/isaaclab/docker/apple-vision-pro`, which lets the conversion job
read the ignored host assets and write its generated USDs back to the host.

## 4. Prepare the scene assets

Still on robolab1:

```bash
cd "$ISAACLAB_ROOT"
./docker/apple-vision-pro/snap-circuits/setup-assets.sh
```

The script performs the following deterministic steps:

- unpacks both ZIP bundles;
- checks out the Sharpa model repository at
  `6eea427eb24189519f32b9f21674cd534d3f973c`;
- checks out Proception's `pro-models` repository at
  `eb8bd682d1ab1a40b8dfbd9d293665165d5519ce` and generates retargeting-only
  URDF copies containing the official MJCF fingertip frames;
- converts the curated OBJ/STL/FBX sources to USD in one headless Isaac Sim
  process;
- scales the downloaded ZIP meshes from millimeters to meters;
- gives movable parts convex collision and small rigid-body masses;
- leaves the SC-100 base grid and tube rack as static colliders; and
- writes `assets/prepared/snap_circuits_table.usda`.

To inspect every discovered object instead, rerun with `--asset-set catalog`.
That overwrites the prepared table with the full 45-object catalog; rerun the
default command to restore the curated bench.

If rclone is configured on robolab1, the complete command is:

```bash
./docker/apple-vision-pro/snap-circuits/setup-assets.sh \
  --rclone-config "$HOME/.config/rclone/rclone.conf" \
  --rclone-source \
    'lance:ai-lumalabs-datasets-ap-se-2-lance/Opal_Sim/shared/assets/objects/snap_circuits/'
```

Private-S3 assets are assumed to use meters. If they use millimeters, repeat
the preparation with `--s3-scale 0.001`.

## 5. Launch the GR1 demo

Stop any existing teleoperation task, then start the proven GR1 task:

```bash
cd "$ISAACLAB_ROOT"
./docker/apple-vision-pro/snap-circuits/start-demo.sh stop
./docker/apple-vision-pro/snap-circuits/start-demo.sh --hand gr1 start
./docker/apple-vision-pro/snap-circuits/start-demo.sh status
```

To use ProHand instead:

```bash
./docker/apple-vision-pro/snap-circuits/start-demo.sh --hand prohand restart
./docker/apple-vision-pro/snap-circuits/start-demo.sh --hand prohand status
```

`ISAACLAB_AVP_HAND=prohand` is equivalent to passing `--hand prohand`.

The AVP client does not choose the backend. It connects to whichever task is
running on robolab1, so switching hands is an explicit server-side restart as
shown above; reconnect the same AVP app after the new server is ready.

ProHand licensing matters: the public model descriptions are BSD-3-Clause,
but the mesh geometry is evaluation/simulation-only under Proception's
separate `MESHES-LICENSE`. The setup script fetches the pinned repository into
ignored local storage; do not commit, redistribute, or bake those meshes into
an image without written permission from Proception.

Wait until status reports:

```text
teleop=running
cloudxr_signaling=listening tcp/48010
```

To inspect startup failures:

```bash
./docker/apple-vision-pro/snap-circuits/start-demo.sh logs
```

### Optional Mac spectator view

Start either hand backend with the fixed 1280x720 H.264 spectator camera:

```bash
./docker/apple-vision-pro/snap-circuits/start-demo.sh \
  --hand gr1 --spectator restart
```

This keeps CloudXR for the AVP and adds a camera-scoped RTSP feed for the Mac.
Open it in VLC (**File > Open Network**) or `ffplay`:

```bash
ffplay -fflags nobuffer -flags low_delay \
  rtsp://172.16.40.15:8554/snap-circuits
```

Allow `8554/tcp` through the robolab1 firewall. The extra camera consumes an
additional render product and NVENC stream; omit `--spectator` when nobody is
watching. This uses Isaac Sim 6's `isaacsim.streaming.rtsp` camera writer, not
the mutually exclusive whole-viewport WebRTC livestream mode.

Do not add `--visualizer none` to the underlying Isaac Lab command. XR normally
auto-injects the headless Kit visualizer, whose per-frame `forward()` and
`app.update()` calls synchronize PhysX Fabric transforms into both CloudXR and
the RTSP render product. Disabling it produces a correctly aligned but frozen
spectator stream.

### Headset-free Isaac Lab preview

Use desktop mode to arrange the scene and verify rendering without wearing the
Vision Pro. It loads the same task and assets, drives the two GR1 wrists and
finger joints with a slow deterministic motion, and publishes the same RTSP URL:

```bash
./docker/apple-vision-pro/snap-circuits/start-demo.sh \
  --hand gr1 --mode desktop --spectator restart
```

Open `rtsp://172.16.40.15:8554/snap-circuits` on the Mac. Both wrists and fingers
must move in this view; a static image means the renderer synchronization is
broken. When scene work is complete, switch to physical hand tracking:

```bash
./docker/apple-vision-pro/snap-circuits/start-demo.sh \
  --hand gr1 --mode avp --spectator restart
```

Desktop mode validates Isaac Lab, physics, retargeted action layout, and RTSP.
Only the final tracked-palm alignment and real hand skeleton require the headset.

### Current interaction scope

The S3 parts retain their authored rigid bodies and detailed stud/bore
colliders. This PR does not add or validate board latching, magnetic snapping,
fixed joints, electrical behavior, or circuit completion. Treat the current
demo as a grasp-and-arrange workbench; a part placed on the grid is not
guaranteed to remain attached.

## 6. Connect the Vision Pro

1. Put the Vision Pro and robolab1 on a network that permits the CloudXR ports
   listed in the parent [Apple Vision Pro runbook](../README.md).
2. Open the signed `CloudXRNativeClient` app on Vision Pro.
3. Enter robolab1's reachable IPv4 address (validated as `172.16.40.15` on the
   lab network).
4. Connect, choose **Start AR**, and grant hand-tracking access.
5. Move both hands above the virtual packing table. The rendered robot hands
   should be the original GR1 hands and their fingers should follow yours.

## Troubleshooting

### The task reports a missing USD

Run `setup-assets.sh` again and verify:

```bash
test -f docker/apple-vision-pro/snap-circuits/assets/prepared/snap_circuits_table.usda
test -f \
  docker/apple-vision-pro/snap-circuits/assets/sharpa-urdf-usd-xml/wave_01/dual_sharpa_wave/dual_sharpa_wave.usda
test -f docker/apple-vision-pro/snap-circuits/assets/pro-models/assets/usd/gen_1_D_L/gen_1_D_L.usda
test -f docker/apple-vision-pro/snap-circuits/assets/pro-models/assets/meshes/prohand_left_with_tips.urdf
```

### The S3 copy is denied

The bucket is private. Confirm rclone is using the intended config and remote:

```bash
rclone config file
rclone listremotes
rclone lsf 'lance:ai-lumalabs-datasets-ap-se-2-lance/Opal_Sim/shared/assets/objects/snap_circuits/' --max-depth 1
```

### Hands are rotated incorrectly

The task uses the Sharpa/IsaacTeleop wrist-frame offsets in
`snap_circuits_sharpa_env_cfg.py`. First use **Reset Origin** in the AVP client
while facing the table. If the physical mounting convention differs, tune the
left/right `target_offset_roll`, `target_offset_pitch`, and
`target_offset_yaw` values there.

### Components are the wrong size

The two ZIP bundles explicitly declare millimeters and are always scaled by
`0.001`. Use `--s3-scale 0.001` only when the private S3 meshes are also in
millimeters.
