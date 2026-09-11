# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera mounts and optics for the Unitree H2 + Sharpa Wave embodiment."""

from __future__ import annotations

from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from .metadata import CAMERA_BY_NAME


# Camera names are defined in metadata; mounts and optics live here.
@configclass
class CameraBaseCfg:
    """Build pinhole or calibrated-fisheye camera configs."""

    @classmethod
    def get_camera_config(
        cls,
        prim_path: str = "/World/envs/env_.*/Robot/d435_link/front_cam",
        update_period: float = 0.02,
        height: int = 480,
        width: int = 640,
        focal_length: float = 7.6,
        focus_distance: float = 400.0,
        f_stop: float = 0.0,
        horizontal_aperture: float = 20.0,
        clipping_range: tuple[float, float] = (0.1, 1.0e5),
        pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rot_offset: tuple[float, float, float, float] = (0.5, -0.5, 0.5, -0.5),
        data_types: list[str] | tuple[str, ...] | None = None,
        camera_type: str = "pinhole",
        **kwargs,
    ) -> CameraCfg:
        if data_types is None:
            data_types = ("rgb",)

        if camera_type == "pinhole":
            spawn = sim_utils.PinholeCameraCfg(
                focal_length=focal_length,
                focus_distance=focus_distance,
                horizontal_aperture=horizontal_aperture,
                clipping_range=clipping_range,
            )
        elif camera_type == "fisheye":
            spawn = sim_utils.FisheyeCameraCfg(
                projection_type="fisheyePolynomial",
                focal_length=focal_length,
                focus_distance=focus_distance,
                f_stop=f_stop,
                horizontal_aperture=horizontal_aperture,
                vertical_aperture=kwargs.get("vertical_aperture"),
                clipping_range=clipping_range,
                # Calibration resolution is independent of render resolution.
                # Keeping the original nominal size lets RTX downsample the
                # calibrated projection without moving its optical centre.
                fisheye_nominal_width=kwargs.get("fisheye_nominal_width", width),
                fisheye_nominal_height=kwargs.get("fisheye_nominal_height", height),
                fisheye_optical_centre_x=kwargs.get("fisheye_optical_centre_x"),
                fisheye_optical_centre_y=kwargs.get("fisheye_optical_centre_y"),
                fisheye_max_fov=kwargs.get("fisheye_max_fov", 196.0),
                fisheye_polynomial_a=kwargs.get("fisheye_polynomial_a", 0.0),
                fisheye_polynomial_b=kwargs.get("fisheye_polynomial_b", 0.00245),
                fisheye_polynomial_c=kwargs.get("fisheye_polynomial_c", 0.0),
                fisheye_polynomial_d=kwargs.get("fisheye_polynomial_d", 0.0),
                fisheye_polynomial_e=kwargs.get("fisheye_polynomial_e", 0.0),
                fisheye_polynomial_f=kwargs.get("fisheye_polynomial_f", 0.0),
            )
        else:
            raise ValueError(f"Invalid camera type: {camera_type}")
        return CameraCfg(
            prim_path=prim_path,
            update_period=update_period,
            height=height,
            width=width,
            data_types=list(data_types),
            spawn=spawn,
            offset=CameraCfg.OffsetCfg(pos=pos_offset, rot=rot_offset, convention="opengl"),
        )


@dataclass(frozen=True)
class CameraMount:
    """Camera mount and optics."""

    prim_path: str
    camera_type: str  # pinhole or fisheye
    focal_length: float
    horizontal_aperture: float
    pos_offset: tuple[float, float, float]
    rot_offset: tuple[float, float, float, float]
    # Fisheye-only fields.
    fisheye_nominal_width: float = 640.0
    fisheye_nominal_height: float = 480.0
    f_stop: float = 0.0
    vertical_aperture: float | None = None
    fisheye_optical_centre_x: float | None = None
    fisheye_optical_centre_y: float | None = None
    fisheye_max_fov: float = 196.0
    fisheye_polynomial_a: float = 0.0
    fisheye_polynomial_b: float = 0.00245
    fisheye_polynomial_c: float = 0.0
    fisheye_polynomial_d: float = 0.0
    fisheye_polynomial_e: float = 0.0
    fisheye_polynomial_f: float = 0.0
    clipping_range: tuple[float, float] = (0.1, 1.0e5)


CAMERA_MOUNTS: dict[str, CameraMount] = {
    "front_camera": CameraMount(
        prim_path="/World/envs/env_.*/Robot/head_yaw_link/head_camera",
        camera_type="fisheye",
        focal_length=0.2667,
        horizontal_aperture=0.576,
        vertical_aperture=0.432,
        fisheye_optical_centre_x=313.66428,
        fisheye_optical_centre_y=238.71598,
        fisheye_max_fov=140,
        fisheye_polynomial_b=3.389688850346405e-3,
        fisheye_polynomial_c=-3.917026323653134e-7,
        fisheye_polynomial_d=7.433522068283752e-9,
        fisheye_polynomial_e=-2.3148518556166975e-11,
        fisheye_polynomial_f=5.110486752596073e-14,
        # Unitree H2 URDF camera mount; rpy=(0, 0, 0) converted to OpenGL xyzw.
        pos_offset=(0.08667, 0.03, 0.0099),
        rot_offset=(0.5, -0.5, -0.5, 0.5),
    ),
    "left_wrist_camera": CameraMount(
        prim_path="/World/envs/env_.*/Robot/left_hand_C_MC/left_wrist_camera",
        camera_type="fisheye",
        focal_length=0.15435,
        horizontal_aperture=0.576,
        vertical_aperture=0.432,
        fisheye_optical_centre_x=319.90624,
        fisheye_optical_centre_y=239.67252,
        fisheye_max_fov=196,
        fisheye_polynomial_b=5.788058552990647e-3,
        fisheye_polynomial_c=1.6881056765838268e-6,
        fisheye_polynomial_d=-4.2228624221772085e-8,
        fisheye_polynomial_e=1.4575948452911756e-10,
        fisheye_polynomial_f=-1.4645006973842839e-13,
        clipping_range=(0.01, 1.0e5),
        # Position from the CAD bracket (支架10°-0506-2.STEP): the camera seat is the
        # 30.8x31.7 mm pad at the end of the bracket arm, 82.8 mm off the hand axis.
        # This is the pad centre -- the sensor's optical centre sits a few mm further
        # along the view axis, which is not modelled here.
        # z tracks the spacer edits to the hand link (2026-07-31): the link went out
        # 12.198181 mm, then back 11.090558 mm, net +1.107623 mm along its local +Z.
        # The bracket keeps its world position through both, so this offset -- measured
        # from the hand link -- absorbs the opposite amount:
        #     -0.0213756 - 0.012198181 + 0.011090558 = -0.0224832
        pos_offset=(0.07498591, 0.0007267178, 0.004823103),
        # Orientation kept from the previous hand-eye calibration.
        rot_offset=(-0.6486821, 0.6996208, 0.1177079, 0.2754761),
    ),
    "right_wrist_camera": CameraMount(
        prim_path="/World/envs/env_.*/Robot/right_hand_C_MC/right_wrist_camera",
        camera_type="fisheye",
        focal_length=0.15435,
        horizontal_aperture=0.576,
        vertical_aperture=0.432,
        fisheye_optical_centre_x=319.90624,
        fisheye_optical_centre_y=239.67252,
        fisheye_max_fov=196,
        fisheye_polynomial_b=5.788058552990647e-3,
        fisheye_polynomial_c=1.6881056765838268e-6,
        fisheye_polynomial_d=-4.2228624221772085e-8,
        fisheye_polynomial_e=1.4575948452911756e-10,
        fisheye_polynomial_f=-1.4645006973842839e-13,
        clipping_range=(0.01, 1.0e5),
        # Mirror of the left mount across the hand's XZ plane, then rolled 180 deg
        # about the camera's own view axis. Valid because left_hand_C_MC and
        # right_hand_C_MC are themselves mirror-defined (measured 0.0000 deg /
        # 0.000000 m from the mirror at the zero pose).
        #
        # The roll is what makes it a real camera rather than a mirror image: a bare
        # mirror flips handedness, so the right image came out upside down (its "up"
        # axis pointed [+0.9953, +0.0708, +0.0662] against the left's [-0.9953,
        # +0.0708, -0.0662]). Rz(180) about the optical axis leaves the view
        # direction untouched (0.0000 deg) and only rotates the image.
        # Mirrors the left camera's spacer compensation; see the note there.
        pos_offset=(0.07964737, 0.001976802, 0.01021968),
        # = mirror(left) * Rz(180 deg). The earlier hand-tuned (-0.6830127, 0.6830127,
        # -0.1830127, -0.1830127) sat 8.6 deg from this, and the bare mirror without
        # the roll sat 180 deg from it.
        rot_offset=(0.6884326, -0.664184, 0.2683462, 0.1136246),
    ),
}
assert set(CAMERA_MOUNTS) == set(CAMERA_BY_NAME), (
    f"CAMERA_MOUNTS {set(CAMERA_MOUNTS)} out of sync with metadata cameras {set(CAMERA_BY_NAME)}"
)


def _camera_cfg_from_mount(mount: CameraMount, **overrides) -> CameraCfg:
    params: dict = dict(
        prim_path=mount.prim_path,
        camera_type=mount.camera_type,
        focal_length=mount.focal_length,
        horizontal_aperture=mount.horizontal_aperture,
        clipping_range=mount.clipping_range,
        pos_offset=mount.pos_offset,
        rot_offset=mount.rot_offset,
    )
    if mount.camera_type == "fisheye":
        params.update(
            f_stop=mount.f_stop,
            vertical_aperture=mount.vertical_aperture,
            fisheye_nominal_width=mount.fisheye_nominal_width,
            fisheye_nominal_height=mount.fisheye_nominal_height,
            fisheye_optical_centre_x=mount.fisheye_optical_centre_x,
            fisheye_optical_centre_y=mount.fisheye_optical_centre_y,
            fisheye_max_fov=mount.fisheye_max_fov,
            fisheye_polynomial_a=mount.fisheye_polynomial_a,
            fisheye_polynomial_b=mount.fisheye_polynomial_b,
            fisheye_polynomial_c=mount.fisheye_polynomial_c,
            fisheye_polynomial_d=mount.fisheye_polynomial_d,
            fisheye_polynomial_e=mount.fisheye_polynomial_e,
            fisheye_polynomial_f=mount.fisheye_polynomial_f,
        )
    params.update(overrides)
    return CameraBaseCfg.get_camera_config(**params)


@configclass
class CameraPresets:
    """Camera presets built from ``CAMERA_MOUNTS``."""

    @classmethod
    def for_camera(cls, name: str, **overrides) -> CameraCfg:
        return _camera_cfg_from_mount(CAMERA_MOUNTS[name], **overrides)

    # Fisheye defaults and pinhole variants.
    @classmethod
    def h2_front_fisheye_camera(cls, **overrides) -> CameraCfg:
        return cls.for_camera("front_camera", **overrides)

    @classmethod
    def h2_front_camera(cls, **overrides) -> CameraCfg:
        return cls.for_camera("front_camera", **{"camera_type": "pinhole", **overrides})

    @classmethod
    def left_shf3l_fisheye_camera(cls, **overrides) -> CameraCfg:
        return cls.for_camera("left_wrist_camera", **overrides)

    @classmethod
    def left_sharpa_wrist_camera(cls, **overrides) -> CameraCfg:
        return cls.for_camera("left_wrist_camera", **{"camera_type": "pinhole", **overrides})

    @classmethod
    def right_shf3l_fisheye_camera(cls, **overrides) -> CameraCfg:
        return cls.for_camera("right_wrist_camera", **overrides)

    @classmethod
    def right_sharpa_wrist_camera(cls, **overrides) -> CameraCfg:
        return cls.for_camera("right_wrist_camera", **{"camera_type": "pinhole", **overrides})
