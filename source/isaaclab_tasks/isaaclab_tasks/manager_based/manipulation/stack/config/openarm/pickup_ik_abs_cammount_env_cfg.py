# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Camera-mount variant of the OpenArm pick-up task.

Identical to `pickup_ik_abs_env_cfg.OpenarmPickUpRedCubeEnvCfg` (same cube/pad geometry, same
action space, same IK/gripper setup, same Mimic subtask signals) except it swaps in
`OPENARM_BI_CAMMOUNT_HIGH_PD_CFG` -- the variant of the OpenArm robot with the real camera-mount
bracket geometry baked into its wrist meshes (see isaaclab_assets/robots/openarm_cammount.py's
docstring for how that URDF was adapted). Exists to close a visual sim2real gap: the real robot
used for VLA data collection has this mount visible in its own wrist/body camera frames, but
every existing task's sim robot doesn't, so a visuomotor policy trained on the old sim robot
never saw its own arm looking like this.

This does NOT touch or modify OpenarmPickUpRedCubeEnvCfg or any file it depends on -- the
existing Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0 task and its Mimic variant are unaffected.

Link name substitutions needed here (the new URDF's LINK names were left as-is during the joint
rename -- see openarm_cammount.py -- so anything referencing a stock-robot link name by string
needs the corresponding new-robot link name):
  openarm_left_link1  -> L_link1     (ee_frame's FrameTransformer source anchor)
  openarm_body_link   -> chest_link  (body_cam mount parent)
  openarm_left_ee_tcp / openarm_right_ee_tcp -> unchanged (added under these exact names when the
    URDF was adapted, specifically so this substitution would NOT be needed for the wrist cameras
    or the IK action terms' body_name).

Camera offsets (wrist_cam/right_wrist_cam relative to ee_tcp, body_cam relative to chest_link)
are inherited UNCHANGED from the stock-robot config as a starting point, not yet retuned for
this robot's different mesh geometry -- the stock offsets were themselves manually tuned in
Isaac Sim against the old mesh shapes (see stack_ik_abs_visuomotor_env_cfg.py's comments), and
this new mount geometry will likely need the same manual retuning. Do that in Isaac Sim, then
copy the resulting pos/rot here, the same way the original values were captured.

Mimic task: Isaac-PickUp-RedCube-OpenArm-CamMount-IK-Abs-Mimic-v0 (not yet created -- add once
this base task is confirmed working, following the same pattern as pickup_ik_abs_env_cfg.py's
own Mimic wrapper).
"""

from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.openarm_cammount import OPENARM_BI_CAMMOUNT_HIGH_PD_CFG

from . import pickup_ik_abs_env_cfg


@configclass
class OpenarmPickUpRedCubeCamMountEnvCfg(pickup_ik_abs_env_cfg.OpenarmPickUpRedCubeEnvCfg):
    """OpenArm pick-up task using the camera-mount robot variant."""

    def __post_init__(self):
        super().__post_init__()  # everything from the stock-robot task, unchanged

        # ── Swap in the camera-mount robot ───────────────────────────────────
        self.scene.robot = OPENARM_BI_CAMMOUNT_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ── Re-point link-name references at the new robot's (unrenamed) link names ──
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/L_link1",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_left_ee_tcp",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                )
            ],
        )
        self.scene.body_cam.prim_path = "{ENV_REGEX_NS}/Robot/chest_link/body_cam"
