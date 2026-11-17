# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the FoveatedCamera sensor."""

from __future__ import annotations

from fovi.sensing.projection import CameraCalibration

from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_fovi.sensors.native_grid import validate_radius_norm


@configclass
class FoveatedCameraCfg(CameraCfg):
    """Configuration for a foveated camera sensor.

    Extends :class:`CameraCfg` with CMF-based foveation parameters. The retina
    looks through a physical camera and places its samples by visual angle
    inside that camera's field of view, densest at the fixation. The rays come
    from :class:`fovi.sensing.coords.SamplingCoords`; ``height`` and ``width``
    are computed from the number of samples.

    :class:`~isaaclab_fovi.sensors.FoveatedCamera` selects its Newton or RTX
    implementation from ``renderer_cfg.renderer_type``.
    """

    class_type: type | str = "isaaclab_fovi.sensors:FoveatedCamera"

    # -- Foveation parameters (no img_height/img_width/fixation_size) --------

    cmf_a_deg: float = 0.591
    """Cortical magnification constant *a* of ``M(e) = 1 / (e + a)`` [deg].

    Smaller values give stronger foveation; a value much larger than the field
    of view gives nearly uniform sampling. The default is an estimate from human
    data. It is passed to :class:`fovi.sensing.coords.SamplingCoords` unchanged."""

    camera_model: CameraCalibration | None = None
    """The physical camera the retina looks through, as :class:`fovi.sensing.projection.CameraModel` fields.

    The retinal window spans this camera's image, and :attr:`mask_to_sensor_fov`
    masks against its edge. The same fields configure fovi's software sampler on
    a real camera.

    If None, the camera that :attr:`spawn` creates is used: an OpenCV model in
    ``spawn.distortion`` (an :class:`~isaaclab.sim.OpenCvFisheyeDistortionCfg` with
    zero coefficients is an ideal equidistant fisheye), otherwise an ideal pinhole
    with the spawn's focal length and apertures. A prim without spawn
    configuration needs a camera model."""

    foveated_res: int = 60
    """Resolution parameter passed to :class:`SamplingCoords`.  For isotropic
    sampling this is the number of radial rings; the total sample count *N*
    depends on ``(fov, cmf_a, foveated_res)``."""

    foveated_style: str = "warped_cartesian_as_grid"
    """Sampling style passed to :class:`SamplingCoords`.

    The default ``"warped_cartesian_as_grid"`` publishes an upright image of
    shape ``(B, H, W, C)``. Sample-set styles such as ``"isotropic"`` are an
    advanced option and publish ``(B, N, 1, C)``."""

    fov_type: str = "circular"
    """Visual footprint; circular is the default for native warped grids."""

    radius_norm: float = 2.0
    """Warped-Cartesian shell norm: 2.0 for circular shells, infinity for square shells."""

    isotropic_plotting_type: str = "v1like"
    """Plotting-space mapping passed to :class:`SamplingCoords` for isotropic
    sampling. Must match the resolved retinal transform so the camera rays
    and policy-side coordinate grids agree on cortical-space geometry. Supported values:
    ``"v1like"`` (default in :class:`SamplingCoords`), ``"schwartz"``, ``"warp"``.
    Ignored for non-isotropic ``foveated_style``."""

    image_rotation_deg: float = 0.0
    """Rotation (degrees, counter-clockwise) applied to the foveated ray
    directions around the optical axis BEFORE rendering.  Used to undo a
    physical camera mount rotation (e.g. an upside-down right-eye sensor)
    so that the per-ray samples land in the canonical
    :class:`SamplingCoords` order — i.e. sample ``i`` always corresponds to
    visual-field position ``rel_cart_coords[i]`` regardless of how the
    camera prim is oriented in world.  General-purpose alternative to
    post-render ``torch.flip`` (which would shuffle the sample order and
    misalign with retinal scatter and KNN processing).

    Common values:
        * ``0.0``    — no rotation (default; right-side-up sensor).
        * ``180.0``  — flip vertically + horizontally (upside-down sensor).
        * Other angles work in principle but are untested."""

    mask_to_sensor_fov: bool = False
    """Zero the samples that fall outside the physical camera's image at the current gaze.

    With gaze the retina can look past the edge of the physical camera; this
    keeps what the sensor reports to what that camera could have seen."""

    retinal_window_short_side_fraction: float = 1.0
    """Diameter of the retinal window as a fraction of the reference image side."""

    retinal_window_reference_side: str = "short"
    """Image side defining the retinal window diameter: ``short`` or ``long``."""

    # -- Optional retinal pooling (photoreceptor -> RGC two-layer sampling) --

    pool_enabled: bool = False
    """If True, the sensor fires a *highres* (photoreceptor) ray grid and
    pools it down to the regular foveated grid. Default False keeps the single-
    layer behaviour unchanged."""

    pool_res_mult: float = 2
    """Resolution multiplier for the highres photoreceptor ray grid.  The
    highres :class:`SamplingCoords` uses
    ``res = round(pool_res_mult * matched_res)`` with the **same** ``cmf_a``
    as the lowres (output) grid so both layers share the same cortical
    magnification profile.  Fractional values are allowed — e.g.
    ``pool_res_mult = sqrt(2)`` gives ~2× input-to-output sample ratio, since
    total sample count scales with ``res²`` in isotropic sampling."""

    pool_k: int | None = None
    """Pooling neighborhood size. ``_as_grid`` sensors use 2D pooling, where
    ``pool_k`` is the squared kernel side (a perfect square) and None uses the
    squared high/low resolution ratio. Sample-set sensors use KNN, where None
    uses the rounded sample-count ratio."""

    pool_mode: str = "avg"
    """Retinal pooling mode. ``"avg"`` or ``"gaussian"``."""

    pool_sample_cortex: bool | str = "geodesic"
    """Space in which KNN neighbours are selected.  ``True`` = Euclidean on
    cortical manifold, ``False`` = visual field, ``"geodesic"`` = exact
    geodesic cortical distances (slower at init, most faithful)."""

    pool_gauss_sigma: float = 10.0
    """Gaussian sigma for ``pool_mode="gaussian"``.  Unused otherwise."""

    pool_int_mode: str = "max"
    """Pool mode used for outputs whose key name contains ``"segmentation"``
    (``semantic_segmentation``, ``instance_segmentation_fast``,
    ``instance_id_segmentation_fast``).  Averaging segmentation values blurs
    class labels and breaks downstream integer/equality checks, so these
    outputs get a separate pooler with this mode.  Set equal to ``pool_mode``
    to disable this behaviour."""

    # -- RTX-specific parameters (ignored by Newton backend) ----------------

    exit_texture_size: int | None = None
    """Dimension of the exit (inverse) LUT texture (EW = EH) for RTX rendering.
    If ``None`` (default), computed as ``max(128, 4 * tw)`` where ``tw`` is the
    direction texture dimension — this ensures the octahedral inverse map has
    sufficient angular resolution to distinguish individual foveated samples.
    Ignored by Newton backend."""

    # height / width are set automatically in _initialize_impl before the
    # parent allocates buffers.  We default them to 1 here so the MISSING
    # sentinel from CameraCfg is satisfied at config-validation time.
    width: int = 1
    height: int = 1

    # Enable per-frame pose updates so the gaze rotation is applied each step.
    update_latest_camera_pose: bool = True

    def __post_init__(self):
        # Resolves the default renderer and forwards deprecated renderer fields.
        super().__post_init__()
        validate_radius_norm(self.radius_norm, self.foveated_style, self.fov_type)
        # ``spawn`` is a ``PinholeCameraCfg`` / ``FisheyeCameraCfg`` whose
        # ``lock_camera`` field defaults to True.  That writes
        # ``omni:kit:cameraLock=True`` on every spawned prim, which the RTX
        # renderer can interpret as "static camera" and use to skip motion-
        # vector synthesis — breaking DLSS-RR temporal reprojection on dynamic
        # geometry.  Foveated cameras always have a moving fixation pose, so
        # we force the lock off here.
        if self.spawn is not None and getattr(self.spawn, "lock_camera", None):
            self.spawn = self.spawn.replace(lock_camera=False)
