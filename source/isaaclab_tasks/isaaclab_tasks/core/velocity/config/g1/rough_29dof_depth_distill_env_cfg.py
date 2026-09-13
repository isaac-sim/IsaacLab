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
from .rough_29dof_mjalign_env_cfg import G129DofRoughMujocoAlignedEnvCfg
from .rough_29dof_mjlab_env_cfg import (
    _LOCOMOTION_JOINTS,
    G129DofRoughMjlabScaleEnvCfg,
    G129DofRoughMjlabScaleHipKneeEnvCfg,
    G129DofRoughMjlabScaleNoFingersEnvCfg,
)
from .rough_29dof_posture_env_cfg import G129DofRoughHipL2AirTime100EnvCfg
from .rough_29dof_power_env_cfg import G129DofRoughWaist1Power2HipPitchLightEnvCfg
from .rough_29dof_waistonly_env_cfg import G129DofRoughAirTime100WaistWarmupEnvCfg

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

_D435_MOUNT_ROT = (0.9150, 0.0, -0.4035, 0.0)
"""Camera orientation as a ``world``-convention quaternion (w, x, y, z): 47.6 degrees about -Y.

Verified against a closed form rather than against the other simulator. On flat ground the depth of
every pixel follows from the camera height, its pitch and its intrinsics with nothing left to tune,
so ``scratchpad/depth_analytic.py`` computes it and the render is differenced against that. At a
measured camera height of 1.2459 m this value gives **RMS 0.042 m** over the frame -- 1.4% of the
1.12 to 3.00 m range -- and reproduces both endpoints of the analytic row profile.

The previous value, ``(0.4035, 0.0, -0.9150, 0.0)``, is the same rotation 180 degrees away and aims
the camera back into the robot's own torso: every pixel reads 0.00 to 0.17 m. That was invisible for
as long as :attr:`~isaaclab.sensors.CameraCfg.update_latest_camera_pose` was left at its default,
because the frozen pose was computed down a different path and happened to produce a plausible
picture of the ground from wherever the camera had been stranded.

One thing is measured but **not yet resolved**: the rendered rows run bottom-up against the analytic
reference and against MuJoCo, which publishes row 0 as the top of the image. Differencing the frame
as-is gives RMS 1.125 m and row-flipped gives 0.042 m. A flat scene cannot tell a row flip from a
180-degree roll about the optical axis, so which of the two it is has to be settled on an asymmetric
scene before anything is flipped here.
"""

_DEPTH_NOISE = {
    "noise_frac": 0.02,
    "dropout_prob": 0.03,
    "blob_prob": 0.15,
    "blob_radius": 3,
}
"""The D435's error model, as the noisy arms apply it.

``noise_frac`` 0.02 is the sensor's own roughly 2% of range; the error grows with distance rather
than being constant, which is why it is a fraction. ``dropout_prob`` and ``blob_prob`` are the part
that matters more: on hardware a depth frame is full of holes -- reflective floor, dark objects,
edges -- and every one of them arrives as *no return*, which
:class:`~g1_deploy.depth.DepthStack` maps to the far range. Simulation produces none: measured on
this task, 5.3% of pixels sit at the far clip and **every one of them is a real hit past three
metres**, not a miss. A contiguous blob is included alongside the per-pixel dropout because a
convolution averages salt-and-pepper away, where a hole it cannot see through is the thing the
policy actually has to survive.
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

    @staticmethod
    def _corrupt(
        observation: torch.Tensor,
        noise_frac: float,
        dropout_prob: float,
        blob_prob: float,
        blob_radius: int,
    ) -> torch.Tensor:
        """Apply the sensor model, in metres, before the term's clip and scale."""
        out = observation
        if noise_frac > 0.0:
            out = out + torch.randn_like(out) * (out * noise_frac)
        if dropout_prob > 0.0:
            lost = torch.rand_like(out) < dropout_prob
            out = torch.where(lost, torch.full_like(out, _DEPTH_MAX_RANGE), out)
        if blob_prob > 0.0:
            batch, _, height, width = out.shape
            hit = torch.rand(batch, device=out.device) < blob_prob
            if bool(hit.any()):
                cy = torch.randint(0, height, (batch,), device=out.device)
                cx = torch.randint(0, width, (batch,), device=out.device)
                ys = torch.arange(height, device=out.device).view(1, height, 1)
                xs = torch.arange(width, device=out.device).view(1, 1, width)
                patch = ((ys - cy.view(-1, 1, 1)).abs() <= blob_radius) & (
                    (xs - cx.view(-1, 1, 1)).abs() <= blob_radius
                )
                patch = patch & hit.view(-1, 1, 1)
                out = torch.where(patch.unsqueeze(1), torch.full_like(out, _DEPTH_MAX_RANGE), out)
        return out.clamp_(min=0.0)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # Without this a freshly reset environment would keep looking at the terrain its previous
        # episode ended on.
        self._stack.reset(env_ids)

    def __call__(
        self,
        env,
        sensor_cfg: SceneEntityCfg,
        data_type: str,
        noise_frac: float = 0.0,
        dropout_prob: float = 0.0,
        blob_prob: float = 0.0,
        blob_radius: int = 3,
    ) -> torch.Tensor:
        """Stacked depth [m], optionally corrupted the way a D435 corrupts it.

        Args:
            env: Environment the term belongs to.
            sensor_cfg: Which camera to read.
            data_type: Annotator to read from it.
            noise_frac: Standard deviation of the range error, as a fraction of the range. A
                RealSense's error grows with distance rather than being constant, so this is
                proportional rather than absolute.
            dropout_prob: Per-pixel probability of returning nothing. Dropped pixels go to the far
                range, not to zero -- that is what the hardware reports and what
                :class:`~g1_deploy.depth.DepthStack` maps a no-return to.
            blob_prob: Per-frame probability of losing a contiguous patch, which is what a
                reflective or dark surface actually produces. Salt-and-pepper dropout alone is too
                easy: a convolution averages it away, where a hole it cannot see through is the
                thing the policy has to survive.
            blob_radius: Half-width of that patch [px].
        """
        camera = env.scene.sensors[sensor_cfg.name]
        frame = camera.data.output[data_type]
        frame = frame.torch if hasattr(frame, "torch") else frame
        # Rays that hit nothing come back as +inf. Clamping them to the far range says "nothing
        # within three metres", which is what the hardware reports; mapping them to zero would say
        # "something touching the lens".
        frame = torch.nan_to_num(frame, nan=_DEPTH_MAX_RANGE, posinf=_DEPTH_MAX_RANGE)
        # Rotate 180 degrees. The rendered frame comes out upside down *and* mirrored against the
        # real D435 and against MuJoCo, which reproduces it: on flat ground the rendered rows run
        # near-to-far top-to-bottom where the analytic reference and MuJoCo both run far-to-near
        # (correlation +0.97 against -0.99), and an obstacle placed on the robot's left lands on the
        # right of the image. Both axes, so it is a roll about the optical axis rather than a row
        # order -- fixed here rather than in the mount quaternion because the asset's own
        # ``d435_link`` frame has no axis pointing where the real camera looks, so there is no
        # authoritative rotation to correct it to.
        observation = frame.permute(0, 3, 1, 2).flip(-2).flip(-1).contiguous()
        if noise_frac > 0.0 or dropout_prob > 0.0 or blob_prob > 0.0:
            observation = self._corrupt(observation, noise_frac, dropout_prob, blob_prob, blob_radius)
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
        # Without this the sensor reports the pose it was given at initialization and never again:
        # the camera stops following the torso, and the depth image is then *constant for the whole
        # episode*. It is the default, it raises nothing, and the picture still looks like a picture,
        # so it cost this line five trained students before it was caught. The extra cost is a
        # FrameView read per step.
        update_latest_camera_pose=True,
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


@configclass
class G129DofRoughYmsDepthDistillEnvCfg(G129DofRoughMjlabScaleEnvCfg):
    """The depth student under ``yms``'s environment.

    ``yms`` -- the per-joint action scale plus left-right mirror augmentation -- is the best gait
    this line has measured: 0.990 success at sd 0.003, and on flat ground under a pinned straight
    command a pelvis roll of 0.55 degrees against the control's 2.8, an airborne share ratio of
    1.03 to 1.07 against 0.71, and a worst joint-pair asymmetry under a degree against three to six.

    The environment is ``yms``'s own. The mirror augmentation lives in how the teacher was trained,
    not in the environment, so the student does not inherit it -- whether the symmetry survives the
    distillation is a thing to measure on the student rather than assume.
    """

    observations: G129DofRoughAirTime100DepthDistillObservationsCfg = (
        G129DofRoughAirTime100DepthDistillObservationsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        _wire_depth_student(self)


@configclass
class G129DofRoughMjDepthDistillEnvCfg(G129DofRoughMujocoAlignedEnvCfg):
    """The depth student under ``mj``'s environment.

    ``mj`` is ``su`` with every difference against the MuJoCo model that a config or an override
    layer can close: the torso and waist inertials, the hardware torque ceilings, 0.2 N*m of joint
    dry friction, 0.05 of passive damping, and a ground friction of 1.0. Its best seed is the
    cleanest walk this family has produced -- success 0.992, single-stance 0.924, flight 0.006,
    pelvis roll 1.65 degrees, no falls, and it tracks a straight command to within 0.002 m/s
    laterally.

    Two things to hold against it. Only one of three seeds trained at 6000 iterations, so the
    teacher is s44 rather than a representative draw. And the arm's ground friction is *pinned* at
    MuJoCo's 1.0 rather than randomized, which is what makes it the right thing to lockstep against
    MuJoCo and the wrong thing to expect floor robustness from --
    :class:`G129DofRoughMujocoAlignedDREnvCfg` is the variant that randomizes around those values.

    Pair this with ``g1_mj_aligned.usda``: the student's environment has to be the teacher's, and
    that includes the asset.
    """

    observations: G129DofRoughAirTime100DepthDistillObservationsCfg = (
        G129DofRoughAirTime100DepthDistillObservationsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        _wire_depth_student(self)


@configclass
class G129DofRoughYhkDepthDistillEnvCfg(G129DofRoughMjlabScaleHipKneeEnvCfg):
    """The depth student under ``yhk``'s environment.

    ``yhk`` is ``yms``'s action-scale table with hip pitch and knee handed back the blanket 0.5, on
    the same mirror augmentation. It is the arm built to fix what killed ``ymsd`` on hardware: the
    full mjlab table leaves the hip 4.5 times less angular authority than the ankle, and the student
    of it stops stepping below roughly 0.3 m/s and goes over backwards. Measured on flat ground at
    0.8 m/s, three seeds: success 0.881 / 0.945 / 0.966, single-stance 0.794 / 0.911 / 0.834, flight
    under 0.006, pelvis roll 0.62 to 0.77 degrees, no falls.

    As with the others the environment is the teacher's own -- the mirror augmentation lives in how
    the teacher was trained, not in the environment, so whether the symmetry survives distillation
    is measured on the student rather than assumed.
    """

    observations: G129DofRoughAirTime100DepthDistillObservationsCfg = (
        G129DofRoughAirTime100DepthDistillObservationsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        _wire_depth_student(self)


def _add_depth_noise(cfg) -> None:
    """Give the depth term the sensor model, in place.

    Args:
        cfg: Environment configuration whose ``observations.depth.image`` term to corrupt.
    """
    cfg.observations.depth.image.params.update(_DEPTH_NOISE)


@configclass
class G129DofRoughYsuDepthDistillNoisyEnvCfg(G129DofRoughAirTime100WaistWarmupEnvCfg):
    """``ysud``'s environment with the depth camera corrupted the way the hardware corrupts it.

    One variable against :class:`G129DofRoughYsuDepthDistillEnvCfg`: the depth term's noise
    parameters. Everything else -- the teacher, the reward set, the asset, the camera geometry --
    is the same, so a difference between the two students is attributable to the sensor model and
    not to anything else.
    """

    observations: G129DofRoughAirTime100DepthDistillObservationsCfg = (
        G129DofRoughAirTime100DepthDistillObservationsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        from .rough_29dof_standup_env_cfg import _STAND_WEIGHTS, _add_stand_height  # noqa: PLC0415

        _add_stand_height(self, _STAND_WEIGHTS["s1"])
        _wire_depth_student(self)
        _add_depth_noise(self)


@configclass
class G129DofRoughMjDepthDistillNoisyEnvCfg(G129DofRoughMjDepthDistillEnvCfg):
    """``mjd``'s environment with the same sensor model. One variable against its clean twin."""

    def __post_init__(self):
        super().__post_init__()
        _add_depth_noise(self)


@configclass
class G129DofRoughNfDepthDistillEnvCfg(G129DofRoughMjlabScaleNoFingersEnvCfg):
    """The depth student under ``nf``'s environment.

    ``nf`` is ``yms`` with the fourteen finger joints taken out of the action and observation
    spaces, which is the set the hardware actually has -- ``g1_deploy`` maps all fourteen to motor
    index -1. It is the cleanest gait this line has measured, on four seeds rather than three:
    success 0.997 / 0.996 / 0.993 / 1.000, single-stance 0.905 / 0.913 / 0.892 / 0.938, flight at
    most 0.008, pelvis roll 0.74 to 1.49 degrees. The mechanism is not luck -- ``action_rate_l2``,
    ``dof_torques_l2`` and ``dof_acc_l2`` are summed over the whole action or joint vector, so on
    the 43-joint arms most of the term meant to smooth the legs was being spent on fingers.

    The teacher is s45, the seed that reached success 1.000 with the best posture of the four.

    **The action and observation contract differs from every other student on this line.** Action
    dimension is 29 rather than 43 and the proprioception loses 42 values, so the export and the
    deployment stack need the 29-joint ordering rather than ``ysud``'s. That is a change toward the
    robot, not away from it, but it is not a drop-in swap for an existing student.

    The teacher group is scoped to the same joints here. It inherits
    :class:`~isaaclab_tasks.core.velocity.velocity_env_cfg.ObservationsCfg.PolicyCfg` directly
    rather than the parent's already-scoped ``policy`` group, so without this it would hand the
    teacher 43 joints of proprioception where its checkpoint expects 29.
    """

    observations: G129DofRoughAirTime100DepthDistillObservationsCfg = (
        G129DofRoughAirTime100DepthDistillObservationsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        for term in ("joint_pos", "joint_vel"):
            getattr(self.observations.teacher, term).params["asset_cfg"] = SceneEntityCfg(
                "robot", joint_names=list(_LOCOMOTION_JOINTS)
            )
        _wire_depth_student(self)


@configclass
class G129DofRoughYsuDepthDistillEnvCfg(G129DofRoughAirTime100WaistWarmupEnvCfg):
    """The depth student under ``ysu``'s environment.

    ``ysu`` is the same mirror augmentation on the ``su`` base -- the blanket 0.5 action scale
    rather than the per-joint table. It reaches 0.978 success with a pelvis roll of 0.10 degrees and
    an airborne ratio of 1.005 to 1.049, so the symmetry result holds on both bases and is not a
    property of the action scale.

    ``su`` is :class:`G129DofRoughStandUpEnvCfg`; the warm-up parent is used here with the
    pelvis-height penalty added below, which is what ``su`` is.
    """

    observations: G129DofRoughAirTime100DepthDistillObservationsCfg = (
        G129DofRoughAirTime100DepthDistillObservationsCfg()
    )

    def __post_init__(self):
        super().__post_init__()

        from .rough_29dof_standup_env_cfg import _STAND_WEIGHTS, _add_stand_height  # noqa: PLC0415

        _add_stand_height(self, _STAND_WEIGHTS["s1"])
        _wire_depth_student(self)
