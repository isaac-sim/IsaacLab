G1 motion-tracking reference data
================================

Data version: v1
Prepared: 2026-10-31
Prepared by: Henry (@Double7sBurger)
Related PR: https://github.com/isaac-sim/IsaacLab/pull/8878

Sources and licenses
--------------------
All six NPZ files below are modified derivatives of the listed source motions.
The applicable upstream licenses and copyright notices are included in LICENSE/.
The Isaac Lab source-code license does not replace these data licenses.

Released file           Original source file          License
g1_cartwheel.npz        MimicKit: g1_cartwheel.pkl    Apache-2.0
g1_run.npz              MimicKit: g1_run.pkl          Apache-2.0
g1_speed_vault.npz      MimicKit: g1_speed_vault.pkl  Apache-2.0
g1_spinkick.npz         MimicKit: g1_spinkick.pkl     Apache-2.0
g1_walk.npz             MimicKit: g1_walk.pkl         Apache-2.0
g1_cr7_grounded.npz     ASAP: CR7 file linked below   MIT

MimicKit motions
  Copyright 2025, Xue Bin Peng
  License text: LICENSE/MimicKit-Apache-2.0.txt

  The five source PKL files were obtained through Mayankm96/isaac-spinning-up:
  https://github.com/Mayankm96/isaac-spinning-up/tree/fe3ee3c618b436692bb4124ae51623de958100bb/source/motion_tracking/data/motions/pkl

  Each source PKL is byte-identical to the corresponding file under
  MimicKit_Data/motions/g1/ in the official MimicKit data download. The archive
  includes an Apache-2.0 LICENSE at its root; these five files have no separate
  motion-specific license. The official download is linked from:
  https://github.com/xbpeng/MimicKit/blob/fbe1a36e878f036d6bf5521defb8eb3b8ab4170d/README.md#installation

ASAP CR7 motion
  Copyright (c) 2025 ASAP Team
  License text: LICENSE/ASAP-MIT.txt

  The source was obtained directly from LeCAR-Lab/ASAP, revision
  df5320cc47dd8cad97961bdfabfe402dd62ad999:
  https://github.com/LeCAR-Lab/ASAP/blob/df5320cc47dd8cad97961bdfabfe402dd62ad999/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl

Permitted uses
--------------
Both licenses allow use for training, modification, redistribution and commercial
use, subject to their full terms. In particular:

- Apache-2.0: provide the license, retain applicable copyright and attribution
  notices, and prominently identify modifications when redistributing. Preserve
  any applicable upstream NOTICE attribution. The license includes a patent grant
  and its conditions; it does not grant trademark rights.
- MIT: include the original copyright and permission notice in copies or
  substantial portions of the data.

The full license texts govern; both include warranty and liability disclaimers.
These mappings apply to the six files listed here, not to other upstream datasets,
robot assets or policy checkpoints.

Modification notices
--------------------
All six released NPZ files were converted from their source motion formats,
resampled to 50 Hz, and packaged with joint/body names, body trajectories and
velocities for the 29-DOF G1. Clip preparation includes standing transitions and
end holds; cyclic motions are repeated to form longer clips.

g1_cartwheel.npz, g1_run.npz, g1_speed_vault.npz, g1_spinkick.npz and g1_walk.npz
are the converted MimicKit derivatives described above.

g1_cr7_grounded.npz additionally changes root height and leg joint poses to align
the feet with the ground during stance. It is an edited ASAP reference motion.

These changes were prepared for Isaac Lab by Henry (@Double7sBurger). The source
motions are credited to MimicKit and ASAP, and the preprocessing workflow to
Mayankm96/isaac-spinning-up.

Contents and format
-------------------
File                      Frames     Bytes
g1_cartwheel.npz              236    312760
g1_cr7_grounded.npz           297    361584
g1_run.npz                    399    567149
g1_speed_vault.npz            196    245664
g1_spinkick.npz               464    594980
g1_walk.npz                   400    579459

Total NPZ size: 2,661,596 bytes. All clips are sampled at 50 Hz.
Robot: official Unitree G1 g1_29dof.usd, without dexterous hands. Each clip
contains 29 joint trajectories and 30 body trajectories. The robot USD is loaded
separately by the task and is not included in this data package.

Read with numpy.load(path, allow_pickle=False). Let T be the number of frames.

Field             Shape       Meaning / units
fps               (1,)        Sampling frequency, Hz
joint_names       (29,)       Joint names in the stored array order
body_names        (30,)       Body names in the stored array order
joint_pos         (T,29)      Joint positions, rad
joint_vel         (T,29)      Joint velocities, rad/s
body_pos_w        (T,30,3)    Body positions in the reference world frame, m
body_quat_w       (T,30,4)    Body orientations, quaternion order WXYZ
body_lin_vel_w    (T,30,3)    Body linear velocities in the world frame, m/s
body_ang_vel_w    (T,30,3)    Body angular velocities in the world frame, rad/s

Trajectory arrays are float32. Frame i has timestamp i / fps. Use the embedded
names to map array columns to robot joints and bodies. The loader converts
stored WXYZ quaternions to Isaac Lab's XYZW convention.

Distribution and integrity
--------------------------
Distribute the six NPZ files, README.txt, LICENSE/ and SHA256SUMS together.
Keep the filenames and directory structure. For server publication, the proposed
prefix is subject to the administrator's final destination:
  Assets/Isaac/6.2/Isaac/IsaacLab/Motions/G1/v1/

Use a new version directory for future data changes instead of replacing v1.
SHA256SUMS covers the six NPZ files, README.txt and both license texts. From this
directory, run before upload and after download:

  Linux:  sha256sum -c SHA256SUMS
  macOS:  shasum -a 256 -c SHA256SUMS

All nine entries should report OK.
