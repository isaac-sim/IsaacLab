# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Distil the w100-line teacher into a student that sees through a chest depth camera.

The teacher's actor reads a terrain height scan and a base linear velocity, and a real G1 publishes
neither. A proprioception-only student is not the answer either -- without exteroception it is
walking blind, and on stairs that shows. This config gives the student what the hardware can
actually provide: five frames of proprioception plus the depth stream of a chest-mounted RealSense.

**The camera frame is authored here, not read from the asset.** The shipped
``Robots/Unitree/G1/g1.usd`` contains no ``d435_link`` -- searched, zero hits -- so there is nothing
to inherit extrinsics from. Isaac Lab's ``CameraCfg`` creates the prim it is pointed at, which is
how WBC-AGILE mounts its own camera (``prim_path=".../Robot/torso_link/front_cam"`` with a
hand-given offset in ``g1_pick_place_tracking_env_cfg.py``), so the same approach is used here with
the D435's measured mount taken from the URDF-converted 29-DoF description.

Three things have to line up or the run is wrong rather than merely worse:

* **The teacher group must reproduce the observation the teacher was trained on**, term for term
  and in order. Here that is the stock policy group unchanged -- proprioception, height scan and
  base linear velocity, single frame, 328 values.
* **The student's proprioception must stay deployable**: no height scan, no base linear velocity,
  five frames, the 690-value contract ``g1_deploy`` already implements.
* **The depth group must stay rank 4.** ``rsl_rl`` splits a model's observation sets by rank and
  routes 4D groups through a convolutional encoder; folding the image into ``policy`` would flatten
  it and throw the spatial structure away.
"""

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import CameraCfg
from isaaclab.utils.buffers import CircularBuffer
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from .rough_29dof_distill_env_cfg import (
    G129DofRoughAirTime100DistillEnvCfg,
    G129DofRoughAirTime100DistillObservationsCfg,
)
from .rough_29dof_dr_env_cfg import _HISTORY_LENGTH, _HISTORY_TERMS
from .rough_29dof_posture_env_cfg import G129DofRoughHipL2AirTime100EnvCfg
from .rough_29dof_power_env_cfg import G129DofRoughWaist1Power2HipPitchLightEnvCfg

_DEPTH_WIDTH = 64
"""Rendered depth width [px]. Small on purpose -- rendering is the cost of this stage."""

_DEPTH_HEIGHT = 38
"""Rendered depth height [px].

Chosen so the vertical field of view lands on the D435's: Isaac Lab derives the vertical aperture
from the horizontal one and the aspect ratio, so ``2 * atan(tan(87/2) * 38/64) = 58.4 deg`` against
the sensor's 58.
"""

_D435_HFOV_DEG = 87.0
"""Horizontal field of view of the D435 depth stream [deg]."""

_D435_FOCAL_LENGTH = 1.93
"""Focal length of the D435 depth module [mm]. Sets the aperture below; only the ratio matters."""

_D435_MOUNT_POS = (0.0576, 0.0175, 0.4299)
"""Camera position relative to ``torso_link`` [m]: forward, left, up.

Read off the ``torso_link/d435_link`` frame of the URDF-converted 29-DoF description, which is the
same robot the shipped asset describes. The shipped asset simply does not carry the frame.
"""

_D435_MOUNT_PITCH_DEG = 47.6
"""How far the camera is pitched down from horizontal [deg], from the same frame.

Confirmed against Unitree's own documentation, which gives 42.4 degrees between the head's vertical
axis and the depth camera's optical axis -- the same 47.6 degrees below horizontal.
"""

_D435_MOUNT_ROT = (0.4035, 0.0, -0.9150, 0.0)
"""Camera orientation as a ``world``-convention quaternion (w, x, y, z).

Not derived -- measured. The first version of this config used the ``ros`` convention with the
quaternion read off the URDF-converted asset's ``d435_link``, and the rendered image had its
near-to-far axis running horizontally: the camera was effectively looking sideways, and the student
trained on it learned to ignore the input entirely (zeroing the depth changed ``success_rate`` by
0.000). Several attempts to fix it by composing rotations failed because ``quat_w_world`` reports
the world convention while the offset was being given in ``ros``, so every check was reading a
different frame from the one being set.

What this value is checked against, on a near-flat mesh terrain: 94.1% of pixels return a hit
(against 68.8% before) and the row means fall monotonically from 2.69 m at the top of the image to
1.11 m at the bottom (correlation -0.97), which is what a camera aimed at the ground ahead must
produce. The earlier value gave a flat row profile and all its variation across columns.
"""

_DEPTH_FRAME_STACK = 3
"""Depth frames handed to the student as channels.

The teacher reads a height grid centred on the robot -- it sees the ground it is standing on and the
ground behind it. A forward-facing camera never sees either, so one frame cannot carry the teacher's
information no matter how well the student fits it. Stacking lets the student carry terrain it has
already walked over into the moment it needs to step on it.

Frames go on the channel axis so the observation stays rank 4; stacking on a new axis would make it
rank 5, which the CNN model rejects.
"""

_DEPTH_MAX_RANGE = 3.0
"""Depth beyond this is clipped [m].

The D435's usable depth falls off well before its optical limit, and terrain past three metres has
no bearing on where the next footstep goes.
"""


class DepthImageStack(ManagerTermBase):
    """The last :data:`_DEPTH_FRAME_STACK` depth frames, as channels of one rank-4 observation.

    Isaac Lab's own per-term ``history_length`` cannot be used here: it stacks on a new axis,
    yielding ``(B, T, C, H, W)``, and ``rsl_rl`` only recognises ``(B, C, H, W)`` as an image.
    """

    def __init__(self, cfg: ObservationTermCfg, env) -> None:
        super().__init__(cfg, env)
        self._stack = CircularBuffer(
            max_len=_DEPTH_FRAME_STACK, batch_size=env.num_envs, device=env.device, stack_dim=1
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # Without this a freshly reset environment would keep looking at the terrain its previous
        # episode ended on.
        self._stack.reset(env_ids)

    def __call__(self, env, sensor_cfg: SceneEntityCfg, data_type: str) -> torch.Tensor:
        camera = env.scene.sensors[sensor_cfg.name]
        frame = camera.data.output[data_type]
        frame = frame.torch if hasattr(frame, "torch") else frame
        # Rays that hit nothing come back as +inf. Clamping them to the far range says "nothing
        # within three metres", which is what the hardware reports; mapping them to zero would say
        # "something touching the lens".
        frame = torch.nan_to_num(frame, nan=_DEPTH_MAX_RANGE, posinf=_DEPTH_MAX_RANGE)
        observation = frame.permute(0, 3, 1, 2).contiguous()
        self._stack.append(observation)
        return self._stack.stacked.clone()


@configclass
class G129DofRoughAirTime100DepthDistillObservationsCfg(G129DofRoughAirTime100DistillObservationsCfg):
    """The blind-student groups plus the camera. ``policy`` and ``teacher`` are inherited unchanged."""

    @configclass
    class DepthCfg(ObsGroup):
        """The student's exteroception. Rank 4, so ``rsl_rl`` routes it through a CNN encoder."""

        image = ObsTerm(
            func=DepthImageStack,
            params={"sensor_cfg": SceneEntityCfg("depth_camera"), "data_type": "distance_to_image_plane"},
            clip=(0.0, _DEPTH_MAX_RANGE),
            scale=1.0 / _DEPTH_MAX_RANGE,
        )

        def __post_init__(self):
            # No corruption: the student is being taught a mapping here, and the sensor model
            # belongs in the sim-to-real step rather than in the supervision signal.
            self.enable_corruption = False
            self.concatenate_terms = True

    depth: DepthCfg = DepthCfg()


def _add_chest_camera(cfg) -> None:
    """Mount the D435 on ``torso_link`` and render depth through Warp, in place.

    Args:
        cfg: Environment configuration to add the camera to.
    """
    cfg.scene.depth_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/depth_camera",
        offset=CameraCfg.OffsetCfg(pos=_D435_MOUNT_POS, rot=_D435_MOUNT_ROT, convention="world"),
        data_types=["distance_to_image_plane"],
        update_period=0.0,
        # Newton's Warp rasteriser rather than Kit RTX: depth needs no shading, and leaving the
        # renderer unset sends the camera down the Kit path, which fails at startup here with
        # "module 'omni.usd' has no attribute 'get_context'". ``presets=`` can still select
        # isaacsim_rtx without touching this file.
        renderer_cfg=MultiBackendRendererCfg(),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_D435_FOCAL_LENGTH,
            horizontal_aperture=2.0 * _D435_FOCAL_LENGTH * math.tan(math.radians(_D435_HFOV_DEG / 2.0)),
            clipping_range=(0.05, 10.0),
        ),
        width=_DEPTH_WIDTH,
        height=_DEPTH_HEIGHT,
    )
    # Render once per control step. The default render interval is tied to the physics step, which
    # would re-render four times per observation for no benefit.
    cfg.sim.render_interval = cfg.decimation


@configclass
class G129DofRoughAirTime100DepthDistillEnvCfg(G129DofRoughAirTime100DistillEnvCfg):
    """The blind-student config plus a chest depth camera. Teacher: the Robust-Waist arm."""

    observations: G129DofRoughAirTime100DepthDistillObservationsCfg = (
        G129DofRoughAirTime100DepthDistillObservationsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        _add_chest_camera(self)


@configclass
class G129DofRoughAirTime100DepthDistillClEnvCfg(G129DofRoughWaist1Power2HipPitchLightEnvCfg):
    """The depth student under the ``cl`` teacher's own environment.

    A teacher hands out actions for the dynamics it was trained in. ``cl`` -- randomization, the
    stronger push, waist L2, the power penalty and hip pitch at -0.15 -- was trained with
    self-collision off, and the Robust-Waist base has it on, so distilling ``cl`` there would
    supervise the student with a teacher being asked about a robot it has never driven. Same student
    wiring, different physics underneath.
    """

    observations: G129DofRoughAirTime100DepthDistillObservationsCfg = (
        G129DofRoughAirTime100DepthDistillObservationsCfg()
    )

    def __post_init__(self):
        super().__post_init__()

        # The student's deployable group and the teacher's own, as in the Robust-Waist variant.
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        for term in _HISTORY_TERMS:
            obs_term = getattr(self.observations.policy, term)
            obs_term.history_length = _HISTORY_LENGTH
            obs_term.flatten_history_dim = True
        self.observations.teacher.enable_corruption = False

        _add_chest_camera(self)


def _wire_depth_student(cfg) -> None:
    """Give ``cfg`` the deployable student group, an uncorrupted teacher group and the camera.

    Args:
        cfg: Environment configuration to wire, modified in place.
    """
    cfg.observations.policy.base_lin_vel = None
    cfg.observations.policy.height_scan = None
    for term in _HISTORY_TERMS:
        obs_term = getattr(cfg.observations.policy, term)
        obs_term.history_length = _HISTORY_LENGTH
        obs_term.flatten_history_dim = True
    cfg.observations.teacher.enable_corruption = False
    _add_chest_camera(cfg)


@configclass
class G129DofRoughAirTime100DepthDistillW100EnvCfg(G129DofRoughHipL2AirTime100EnvCfg):
    """The depth student under w100's own environment.

    The first run of this line distilled w100 inside the Robust-Waist environment -- domain
    randomization, the stronger push and self-collision -- none of which w100 was trained with. The
    student fitted those actions to a behaviour-cloning loss of 0.024 and still reached only 0.476
    against the teacher's 0.993, because what it was copying was w100 being asked about a robot and
    a disturbance it had never met. Here the physics is w100's, so the teacher is answering
    questions it can answer.
    """

    observations: G129DofRoughAirTime100DepthDistillObservationsCfg = (
        G129DofRoughAirTime100DepthDistillObservationsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        _wire_depth_student(self)
